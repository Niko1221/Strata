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

## C. Where the head runs, and the batch slots beside a layer split (delta 3)

The output head and the MTP draft layer run on the **last stage** of a layer split, and the split search places the
layers - never the head. On a two-card machine that means the head lands on whichever card the pipeline ends on,
which is not necessarily the faster one (`auto` orders by generation, and the head follows). `--head-device` is the
placement of its own; the card **order** is what puts the head on another card.

| Flag | Default | What it does | Scope (when it applies and when it does NOT) | Limit or non-claim | Since |
| --- | --- | --- | --- | --- | --- |
| `--head-device D` (env `STRATA_HEAD_DEVICE`) | `-1` (as placed: the last stage) | The device that runs the output head **and**, with it, the MTP draft layer (the drafter reads the last stage's residual, so the two move together). It names one of the devices the pipeline already uses (`0` is the primary, the rest are `--split-device`): the later stages are reordered so that one is last. | `--serve` with a layer split. With one card it is a no-op (`D` must be `0`). A device that is not one of the pipeline's, or the primary device of a split, is **said and ignored** - the head stays where it was. | It is **not** a layer placement: the split points do not move, only which stage owns the head. It cannot move the head onto the primary device of a split (device 0 runs the first stage and the prompt path; a head-only tail stage is not implemented) - order the cards instead (below). It is **not** a speed claim: nothing measures anything here. | `layer/delta3` |
| `"head_device": N` (config; the server reads it, not the engine) | absent | The card, numbered as `nvidia-smi` numbers them, that must hold the head. The server lists it **last** in `CUDA_VISIBLE_DEVICES`, so the engine's last stage - the head's - is that card. | The server's config (`strata-*.json`), or `setup --head-device N|auto`. Needs two or more cards in `"gpu"`; a card that is not one of them is refused. | It reorders the cards, so an **explicit** `"layer_split"` keeps its meaning *in the new order* ("24" is still "the second card starts at layer 24", but the second card is a different card). No engine flag is involved: an older engine understands it. | `layer/delta3` |
| `--batch-mtp` (env `STRATA_BATCH_MTP=1`) | off | With `--batch N` and the MTP drafter (`--mtp`, `--spec T`), each batch slot also verifies **one** MTP proposal per window. | `--serve`, `--batch 2+`, `--mtp`, `--spec T >= 2`. **A layer split works too (delta 3)**: every stage keeps its own session per slot and the slot drafters live on the last stage's device. Not with `--split-device 0` (both stages on one GPU: the batch slots themselves need a card per stage). Anything else the engine says in one line (`WARNING: --batch-mtp is off: ...`) and batches as usual. | The VRAM it costs is per slot: the drafter's K/V and buffers, on the head's card. Upstream measured +31 % to +39 % total throughput on an RTX PRO 5000 with 2-4 clients **on one GPU**; this tree has **not** measured it under a layer split (docs/BATCHING.md says so). One proposal per slot: more drafts accepted per window is not what this does. The acceptance is **reported** (delta 3's measurement): each slot's `BDONE` ends with `accepted offered`, and the `strata batch:` summary line carries the period's `MTP drafts accepted A of O (P%)` - absent without the flag. **A flag never ends the engine**: if a slot's drafter cannot be built, or the draft head's GGML type has no native MMVQ path, or a draft fails while serving, the engine says `WARNING: --batch-mtp is off: <reason>` and the slots go on decoding **without drafts** (the request is served). | `engine 0.1.39` (upstream); the layer split with it, and the fail-open, are `layer/delta3` |

## D. Upstream flags these are confused with

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

What has been verified against this base (build, flags, tests) and what has not (any live run):
[SPILL_AND_PROMPT_CACHE.md](SPILL_AND_PROMPT_CACHE.md), "Validation environment" and "Non-claims".
