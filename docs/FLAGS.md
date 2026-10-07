# Flags of the disk layer

This page lists every `strata --serve` flag this tree adds, the two functions they belong to, and the upstream
flags they are most often confused with. The names and the defaults are the exact strings the engine parses and
prints in `--help`. A flag the engine does not know is **fatal** (`unknown argument`, then `usage`, exit code 2),
so a config for an older engine must not carry these flags.

The functions themselves are described in [SPILL_AND_PROMPT_CACHE.md](SPILL_AND_PROMPT_CACHE.md). The behaviour in
prose is also in [DETAILS.md](DETAILS.md) ("Disk tier for the parked conversations" and "System-prompt prefill
cache").

## How to read the "Since" column

- **`layer/base`** — this tree's base: upstream `v0.1.40.1` plus PR #1271 (the disk tier) and PR #1269.
- **`layer/delta1`** — this tree's own work on top of that base.
- **`engine <version>`** — the flag came from upstream and is unchanged here.

## Three caches, not one

Do not mix these. They share the word "cache" and little else:

1. **The parked conversations in RAM** (`--conversation-cache-mib`, `--conversation-cache-slots`,
   `--conversation-cache-min-free-mib`; upstream engine). Conversations the engine keeps in host RAM between
   requests, so a returning chat does not read its prompt again. RAM only: lost on restart.
2. **The disk tier** (`--conversation-cache-spill-dir` + `--conversation-cache-disk-mib` + the three flags beside
   them). A conversation the RAM cache evicts is written to a folder as an ordinary session file, and read back
   later or after a restart.
3. **The system-prompt prefill cache** (`--system-prompt-cache*`). The checkpoint root that ends the system prompt
   (`--prompt-cache-root`, in RAM) is persisted to its own folder as a session file and reloaded at start.

Two nearby names are **not** caches: `--kv-resident` decides **where** the context's K/V lives (VRAM or host RAM),
and `slot_save_path` is the server's manual save/restore API (the same session files, written only when a client
asks). The RAM-side prefill checkpoints (`--prompt-cache`, `--prompt-cache-every`, `--prompt-cache-tail`) are what
the system-prompt cache persists; `--prompt-cache-root` is the threshold for the root checkpoint it targets.

## A. The disk tier (conversations on disk)

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--conversation-cache-spill-dir DIR` | empty (off) | Turns the disk tier on and names its folder. A conversation the RAM cache evicts is written there as an ordinary session file (`.sess`) plus a small sidecar (`.meta`). | `--serve` only. Needs `--conversation-cache-mib > 0`, `--prompt-cache > 0`, `--conversation-cache-slots > 0` and a nonzero `--conversation-cache-disk-mib`; otherwise the engine warns and the tier stays off. Works with `--layer-split` (one file per stage). | No authentication: the files hold the conversation's token IDs and state, so the folder must stay private. A file from another model or another configuration (including another KV quant) is **rejected, not reused**. | `layer/base` (`v0.1.40.1` + PR #1271) |
| `--conversation-cache-disk-mib N` | 8192 | The spill folder's byte budget, in MiB. The GC removes stored conversations over it, oldest first. | `--serve`; inside the disk tier. `0` disables the disk tier. This is the tier's own folder, not `--conversation-cache-mib`, which is the RAM budget. | A budget, not a disk reservation or a quota. A stored conversation larger than the whole budget is kept anyway (removing it could not bring the folder under the budget) and counted as `oversized`. | `layer/base` |
| `--conversation-cache-spill-when-full MODE` | `evict-oldest` | What the GC does at the budget. `evict-oldest` drops the oldest stored conversation to make room (what the tier always did). `reject` stores nothing new and removes nothing. | `--serve`; only inside the disk tier. An unknown mode is refused at start. | `reject` does not bound the folder in any other way: a folder already over budget stops accepting spills and keeps what is there. The age lever below is independent of this. | `layer/delta1` |
| `--conversation-cache-spill-max-age-days N` | 0 (off) | Optional age pruning. With a positive N, the GC removes stored conversations older than N days, oldest first, counted separately from the budget GC. | `--serve`; only inside the disk tier. `0` means **no deletion by time at all**: this lever is off unless set. | Off by default. Independent of `--conversation-cache-spill-when-full`: both can act, each with its own counter in the log. | `layer/delta1` |
| `--conversation-cache-similarity F` | 0.0 | The least fraction of the new prompt that a stored (RAM or disk) conversation must share as a common prefix before it may be reused. | `--serve`; applies to the RAM cache and the disk tier's match. A finite number in `[0, 1)`; anything else is refused. | 0 accepts any match. It filters weak hits only: a candidate that passes still has to be an exact token/image prefix to be restored. | `layer/base` |
| `--conversation-cache-n-min N` | 0 | The least number of common-prefix tokens a stored conversation must offer before it may be reused. | `--serve`; same place as `--conversation-cache-similarity` (RAM and disk). | 0 accepts any length. A hit that offers fewer tokens than a live slot or the RAM cache already reaches is not used. | `layer/base` |

The scan never deletes: at start any file it does not understand (another model/config identity, a broken or
missing sidecar, an orphan session file, a leftover temporary) is **ignored and counted**, not removed. Only the GC
removes, by budget and (if enabled) by age, oldest first, with a counter and a log line.

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
