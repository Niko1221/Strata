# Disk park version gate: RTX 3090, IQ3_XXS, 262,144-position context

Measured on **2026-10-12** by [leocoic](https://github.com/leocoic): desktop RTX 3090 (24 GB),
Ubuntu 22.04.5, driver 580.178.04. Engine built from branch `f-disk-park-0143` at `8364dea7`
(`strata 0.1.43`), stock HTTP server under systemd, `--kv k8v4 --max-context 262144`, IQ3_XXS
pack, disk park tier enabled with the stream dir under the reporter's home directory.

This entry does **not** measure speed. A version gate decides which caches a new engine may read;
the question it has to answer is whether gating costs reuse. It did not.

## 1. Snapshot layer: the tag changed, the bytes did not move

| | before | after |
| --- | --- | --- |
| tag directory | `tag-A812110E` — 68 GB, 3,747 session dirs | `tag-AA00AEC7` — same 68 GB, renamed |
| tag algorithm | field table `v1` (model / kv / context / rope) | field table `v2` (same fields + `STRATA_VERSION`) |
| bytes re-stored | — | **0** |

The rename is only valid because the numerics did not change across this upgrade — the gate was
introduced, not the routing. A real version bump opens a fresh directory on its own.

## 2. Reuse right after the restart

First 8 served requests, same conversation, `/metrics?requests=all`:

| metric | value |
| --- | --- |
| `prompt_tokens` | 735,385 |
| `reused` | 644,027 (87.6%) |
| `prompt_read` | 686 – 1,326 |
| `hit_rate` | 0.903 – 0.917 |
| `decode_tok_s` | 118.5 – 127.4 |

`conversation_cache`: 7,168 MiB budget, 8 slots, 1 entry parked, 0 restores needed. Nothing had to
be re-prefilled because the parked tier was renamed rather than invalidated.

## 3. Blob layer: the old file was archived, not relabelled

| | value |
| --- | --- |
| old `checkpoints.bin` | magic `SKB1`, 20,424,930,848 B |
| who wrote it | 0.1.41 until 13:27, 0.1.43 after — one file, two numeric paths |
| what was done | archived as `checkpoints.bin.skb1-<stamp>`, not relabelled |
| new file | magic `SKB2`, header tag `AA00AEC7`, 118,063,200 B after the first `put` (one 118,063,152 B record) |
| entry header | also carries the tag at `@+4` (`od` line 2: `534b 4232 c7ae 00 aa …`) |
| growth measured | 19,362,362,192 B (17:46) → 20,424,930,848 B (18:04) ≈ 3.5 GB/h |

Relabelling the old blob would have been a lie: its records were written by two different engines.
Under the gate the old format is unreadable by design, so the honest options are archive or delete.

## 4. Unit tests on this build

467 checks, 0 failures — blob 68 (new version-gate section) / digest 28 / spool 64 / logs 98 /
session 136 / bridge 54 / image 19. Compiled `g++ -std=c++20 -O1 -I include` against the eight
`src/core/disk_park_*.cpp` units.

## Out of scope here

The 0.1.41 → 0.1.43 decode change on this machine (window medians 116.8 → 122.5 tok/s; long
outputs 1,000–2,500 tokens 94.0 → 123.1) is a **version upgrade** measurement, not an effect of
this gate, and is deliberately not attributed to it.
