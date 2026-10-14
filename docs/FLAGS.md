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
(`evict`) reproduces delta 1 byte for byte. Without `--conversation-cache-spill-dir` both are inert.

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--conversation-cache-spill-dir DIR` | empty (off) | Turns the disk tier on and names its folder. A conversation the RAM cache evicts is written there as an ordinary session file (`.sess`) plus a small sidecar (`.meta`). | `--serve` only. Needs `--conversation-cache-mib > 0`, `--prompt-cache > 0`, `--conversation-cache-slots > 0` and a nonzero `--conversation-cache-disk-mib`; otherwise the engine warns and the tier stays off. Works with `--layer-split` (one file per stage). | No authentication: the files hold the conversation's token IDs and state, so the folder must stay private. A file from another model or another configuration (including another KV quant) is **rejected, not reused**. | `layer/base` (`v0.1.41` + PR #1271) |
| `--conversation-cache-disk-mib N` | 8192 | The spill folder's byte budget, in MiB. The GC removes stored conversations over it, oldest first. | `--serve`; inside the disk tier. `0` disables the disk tier. This is the tier's own folder, not `--conversation-cache-mib`, which is the RAM budget. | A budget, not a disk reservation or a quota. A stored conversation larger than the whole budget is kept anyway (removing it could not bring the folder under the budget) and counted as `oversized`. | `layer/base` |
| `--conversation-cache-spill-when-full MODE` | `evict-oldest` | What the GC does at the budget. `evict-oldest` drops the oldest stored conversation to make room (what the tier always did). `reject` stores nothing new and removes nothing. | `--serve`; only inside the disk tier. An unknown mode is refused at start. | `reject` does not bound the folder in any other way: a folder already over budget stops accepting spills and keeps what is there. The age lever below is independent of this. | `layer/delta1` |
| `--conversation-cache-spill-max-age-days N` | 0 (off) | Optional age pruning. With a positive N, the GC removes stored conversations older than N days, oldest first, counted separately from the budget GC. | `--serve`; only inside the disk tier. `0` means **no deletion by time at all**: this lever is off unless set. | Off by default. Independent of `--conversation-cache-spill-when-full`: both can act, each with its own counter in the log. | `layer/delta1` |
| `--conversation-cache-spill-on MODE` | **`park`** | What the tier is there for. `park` (the default **when the tier is on**: a spill directory and a nonzero budget) also writes a conversation the moment it is parked at the end of a request, so a close loses at most the request in flight. `evict` is the delta-1 behaviour: only what the RAM cache evicts reaches disk. | `--serve`; inside the disk tier. **The default changes with the tier**: without `--conversation-cache-spill-dir` neither mode does anything (no folder, no byte). An unknown mode is refused at start. | The mirror writes at the end of each request, never mid-generation, and collapses a burst of parks of one conversation (one cell per conversation holding only the newest state; the async writer drains it). The disk is bounded by `--conversation-cache-disk-mib` exactly as before. | `layer/delta2` |
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
| `--conversation-cache-mib N` | 0 (off) | The RAM budget for parked conversations. The disk tier is an extension of this RAM cache, not a replacement. | `--serve`. The disk tier needs it nonzero. It bounds host RAM only. | Upstream parking is RAM-only: without `--conversation-cache-spill-dir`, a parked conversation does not survive a restart. | engine 0.1.39 (upstream) |
| `--conversation-cache-slots N` | 4 | At most N parked conversations in RAM. | `--serve`; the RAM cache. The disk tier needs it nonzero. | Not the count of conversations stored on disk: the disk tier's index holds its own set (up to 256), bounded by `--conversation-cache-disk-mib`. | engine 0.1.39 (upstream) |
| `--conversation-cache-min-free-mib N` | 2560 | The physical-RAM floor when parking or restoring a session file. | `--serve`; the RAM cache and every disk hit, which is read into host RAM before it is restored. | A disk hit that does not fit `--conversation-cache-mib`, or does not leave this floor available, is skipped and the prompt is read normally. It is a preflight, not a reservation. | engine 0.1.39 (upstream) |
| `--kv-resident N` | 0 / setup | Where the context's K/V lives: keep N cells of each QSA layer in VRAM and the rest in host RAM (KV streaming, minimum 20480). | `--serve`; the KV cache, not any conversation cache. | **Not a cache and not part of this layer.** It changes the per-turn read cost and VRAM residency; the disk tier does not. | engine 0.1.5 (upstream) |
| `--prompt-cache-root N` | 2048 | The minimum system-prompt length (tokens) at which the first turn boundary is checkpointed as the root, in RAM. | `--serve`; the RAM prefill checkpoints. The system-prompt cache persists this root. `0` = no system-prompt checkpoint. | RAM only by itself: it does not survive a restart. The system-prompt cache is what makes the root durable. | engine 0.1.39 (upstream) |
| `slot_save_path` (config; `--slot-save-path DIR` on the server) | empty (off) | The folder for the manual `POST /slots/0?action=save` and `?action=restore` API. A saved file is the same session format a spill writes. | The server, not the engine. Needs the model loaded, a single slot, and no `--batch`. | An explicit client action, not a cache: nothing is written unless a client asks, and nothing deletes the files. | engine 0.1.40.1 (upstream server) |

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
