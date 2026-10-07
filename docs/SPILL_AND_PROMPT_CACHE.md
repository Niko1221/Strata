# Keeping conversations and the system prompt on disk

Two opt-in functions of `strata --serve` write a checkpoint to disk and read it back instead of reading a prompt
again: the **disk tier** keeps a conversation the RAM cache evicts, and the **system-prompt cache** keeps the
checkpoint that ends the system prompt. Both are off by default. With no flags, the engine behaves exactly like
upstream. Every flag is listed in [FLAGS.md](FLAGS.md).

## Why this helps

A long chat that comes back after a few turns is not free: the engine has to hold, or re-read, the history in
front of the new message. Upstream keeps that in host RAM (`--conversation-cache-mib`), but the RAM cache is
bounded and does not survive a restart. The disk tier closes that gap: when the RAM cache evicts a conversation it
is written to a folder as an ordinary session file, and a later request — or a restart — reads it back instead of
reading the whole prompt again. The file is the same format the manual slot save/restore API writes, so a spilled
conversation and a hand-saved one are interchangeable.

The system prompt in front of every chat is (for one client) the same on every new chat, and it can be thousands
of tokens. Re-reading it costs time on the first request of each chat. The system-prompt cache persists the
checkpoint at the end of that prefix, so a new chat reads only what comes after it. It is a resume point for the
prefix, never a conversation tail.

## Slot lifecycle

A slot is one request's state. When its turn ends the state is parked; the RAM cache holds it, and the disk tier
catches what the RAM cache drops.

```mermaid
flowchart TD
    A["Turn ends: state in VRAM / host RAM"] --> B{"Fits the RAM cache budget?"}
    B -->|yes| C["Parked in RAM"]
    B -->|no| D["Evicted: spill to disk"]
    D --> E["One session file per stage (.sess) + one sidecar (.meta), written to a temp name then renamed"]
    C --> F["Next prefill: match in RAM (no disk read)"]
    E --> G["Next prefill: match from the sidecars only (no K/V read)"]
    G --> H{"Identity and prefix valid?"}
    H -->|no| I["Rejected clean: ignored and counted, never deleted"]
    H -->|yes| J["Validate header and hash, load, grow or compact"]
    J --> K["Re-spill on a later eviction: atomic replacement (temp + rename)"]
    K --> E
    D --> L["GC: the only thing that deletes"]
    L --> M["By space: over --conversation-cache-disk-mib, oldest first"]
    M --> N{"Is the age lever on?"}
    N -->|yes| O["By age too: oldest first"]
    N -->|no| P["No deletion by time at all"]
```

## Prefill decision tree

A new prompt is first offered to the live slots and the RAM cache; only then to the disk tier. A disk match is
loose (a common prefix), and the candidate is restored only if the exact prefix passes; otherwise the prompt is
read normally.

```mermaid
flowchart TD
    A["A prefill arrives"] --> B{"A live slot or the RAM cache reaches it?"}
    B -->|yes| C["Hit in RAM: reused, disk untouched"]
    B -->|no| D["Match on disk, from the sidecars only (no K/V read)"]
    D --> E{"Common prefix passes similarity and n-min?"}
    E -->|no| F["Miss: normal reprocessing"]
    E -->|yes| G{"Model / config identity and KV type match?"}
    G -->|no - another model, another quant, another build| H["Rejected clean: ignored and counted, not reused"]
    H --> F
    G -->|yes| I{"Fits the RAM budget and the min-free floor?"}
    I -->|no| F
    I -->|yes| J["Validate header / hash, load, grow or compact"]
    J --> K["Hit: only the tokens after the prefix are processed"]
```

## System-prompt variants

The cache does not hold one system prompt but a small set of variants. A variant is keyed by the exact hash of
its system-prompt token prefix (plus `--system-prompt-cache-key` and the model/config identity), so only a prompt
that begins with exactly those tokens can attach it.

```mermaid
flowchart TD
    A["New chat"] --> B["Hash its system-prompt token prefix"]
    B --> C{"A variant with that hash exists?"}
    C -->|yes| D["Hit: load the stored root, process only the tokens after it"]
    C -->|no| E["Miss: reprocess from the start and write the new prefix as its own variant"]
    E --> F["The old variant is kept: it is a hit again if the client returns to it"]
    D --> G["GC: the only thing that deletes"]
    F --> G
    G --> H["By space: over --system-prompt-cache-slots / -mib, oldest first"]
    G --> I["By age: --system-prompt-cache-max-age-days, oldest first; 0 = never"]
    H --> J["A variant never dies because the system prompt changed"]
    I --> J
```

## Use and limits

**Turn it on.**

- Disk tier: `--conversation-cache-spill-dir DIR`, with `--conversation-cache-mib > 0`, `--prompt-cache > 0`,
  `--conversation-cache-slots > 0` and a nonzero `--conversation-cache-disk-mib` (default 8192 MiB).
  `--conversation-cache-spill-when-full` chooses `evict-oldest` (default) or `reject`;
  `--conversation-cache-spill-max-age-days` adds optional age pruning (0 = off).
- System-prompt cache: `--system-prompt-cache` with `--system-prompt-cache-dir DIR`. It also needs
  `--prompt-cache > 0`, `--prompt-cache-root > 0`, a turn token and `--mtp` (without MTP the feature reports
  itself off rather than capturing a different artifact). Its folder is separate from the spill folder.

Both can be set in a config's `args` and edited from the Settings page (see `serve/runconfig.py`). The Monitor
shows their counters under `/metrics` in `conversation_cache` (`disk` and `system_prompt_cache`).

**Limits.**

- The disk tier works with `--layer-split`: one session file per stage plus one joint sidecar.
- The system-prompt cache works with `--layer-split` too, the same way: one session file per stage under the
  variant's key, and a joint sidecar holding the stage count (its sidecar is version 2; a version 1 file, the
  single-file form, is still read). It also loads a variant whose K/V was stored while the engine kept part of
  the context in host RAM.
- A disk hit is read into host RAM before it is restored: it must fit `--conversation-cache-mib` and leave the
  `--conversation-cache-min-free-mib` floor available, or the request reads the prompt normally.
- The scan never deletes. A file of another identity, a broken or missing sidecar, an orphan session file and a
  leftover temporary are ignored and counted (`foreign`, `stale`, `orphan`), never removed. Only the GC removes,
  by budget and (if enabled) by age, oldest first.
- Nothing is written with the flags absent: no folder is created and no byte is written.

## Validation environment

- The session-file numbers below were measured on an **RTX 4070 Ti (12 GB), Ryzen 9 5900X, 64 GB RAM, NVMe ext4,
  IQ3_XXS**, on a 63,025-token conversation, with engine 0.1.38 carrying the session-file change (binary sha256
  `3bbe4fc3...`). They measure the **session-file SAVE/RESTORE path**, which the spill reuses; they are not a
  measurement of the automatic spill.
- The KV-streaming figure (~13.7 KB of RAM per context token) is measured on the setups in
  [DETAILS.md](DETAILS.md#speed-measured) (KV streaming, engine 0.1.5; ~1.7 GB at 128K).
- The layer's own host-only tests (conversation cache / memory / file / spill / prompt cache) build and pass with
  `-DSTRATA_BUILD_CONVERSATION_TESTS=ON`. That is a correctness check, not a speed measurement.

## Measured benefit

**Session files (what both functions write).** On the setup above, a 63,025-token conversation is a **1.20 GB**
file: **save 0.65 s**, **restore 1.05 s** in a new engine process, and the next 32-token turn then takes
**1.14 s** (1.00 s for the same turn without a restart). The cold first turn of that conversation takes 25.5 s.
The file reported `n_written = 1,198,691,396` bytes, that is **about 19,019 bytes per token** at this geometry.
Each extra checkpoint is about **118 MB**. These are single runs.

**KV streaming.** Holding the context's K/V in host RAM costs about **13.7 KB per context token** (1.7 GB at
128K). This is the dominant term when sizing a spill folder.

**This layer, end to end.** There is **no measurement yet of the disk tier or the system-prompt cache with the
model loaded**: no hit rate, no tokens saved, no per-turn latency. The counters exist (hits, misses, tokens saved,
bytes, live variants, evictions by space and age, hash changes) and are logged at start and at shutdown; they have
not been read from a real run.

## Sizes by KV type

The KV cache is the part that scales with its type. `int8` is the type measured here; `f16` and `q4_0` are
**derived** from it by the nominal width (f16 = 2 bytes/value, int8 = 1, q4_0 = 18 bytes per 32 values = 0.5625),
so they are estimates, not measurements. The derivation ignores the per-block scale bytes and does not separate
the parts that do not scale with the KV (running state, checkpoints); treat them as a lower bound on the total
disk a folder will use.

Measured `int8` anchors: **~14,029 B (≈13.7 KB) per context token** for the KV in host RAM (streaming), and
**19,019 B per token** for the session file on disk (63,025 tokens → 1.20 GB).

| KV type | KV in host RAM (B/token) | 17k | 63k | 384k | Session file on disk (B/token) | 17k | 63k | 384k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **f16** | ~28,058 *(derived)* | ~455 MiB | ~1.65 GiB | ~10.0 GiB | ~38,038 *(derived)* | ~617 MiB | ~2.23 GiB | ~13.6 GiB |
| **int8** | ~14,029 *(measured)* | ~227 MiB | ~0.82 GiB | ~5.0 GiB | 19,019 *(measured)* | ~308 MiB | 1.20 GB *(measured at 63,025)* | ~6.80 GiB |
| **q4_0** | ~7,891 *(derived)* | ~128 MiB | ~0.46 GiB | ~2.8 GiB | ~10,698 *(derived)* | ~173 MiB | ~0.63 GiB | ~3.83 GiB |

Sizes are powers of 1024 (MiB, GiB) except the measured 1.20 GB, which is the decimal figure upstream reports.

## Non-claims

- **Attention is causal.** A token's K/V was computed against the system prompt that was in front of it, so the
  tail of a conversation (its K/V) cannot be kept under a _different_ system prompt: everything after the
  divergence is **reprocessed**. If the change is at the **end** of the system prompt, only the tail is paid; if it
  is at the **beginning**, everything after it is paid.
- A file from **another model or another configuration** (including **another KV quant**) is **rejected**, not
  reused. The same identity check that guards the slot save/restore API guards the spill.
- The disk tier **does not speed up the path that already reuses**. It does not fix the per-turn cost and it does
  not change VRAM residency; those are `--kv-resident` and the slot count (`parallel`). The disk tier only avoids
  re-reading a prompt that would otherwise be read again.
- There is a real, documented case in the field in which a client that **mutates the head of the prompt** (a
  per-request attribution/version block) destroys reuse: the system-prompt prefix hash changes every request, so
  every request is a miss. The cache does not fix that; it makes it visible (a `hash_changes` counter).
- **There is no end-to-end measurement yet** with the model loaded. The hit rate, the tokens saved and the
  per-turn latency of these two functions have not been read from a real run.
