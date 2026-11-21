[Engine] Persist the SSD conversation tier across restarts when the model fingerprint matches

## Problem and result

The SSD conversation tier (`--conversation-spill-dir`, on by default beside the data pack) parks
spilled conversations as session files, but its folder belongs to one engine process: it is locked
per PID and removed when the engine exits — including on a restart or a reboot. Every long-lived
conversation then pays full prompt re-read cost on the first turn after the restart, even though the
model and settings are unchanged and the parked state on disk was still exactly resumable.

With `--conversation-spill-persist` the tier lives in a stable folder named by the model's
fingerprint — `<root>/fp-<16 hex>` — and its entries survive the process. At start the engine takes
the folder (a lock file; a second live engine of the same model falls back to its own per-process
folder as before), reads each entry's small index file, and adopts every entry whose recorded
identity (model fingerprint + config fingerprint) matches this run: no session-file read is needed
to index them. Entries of another fingerprint, half-written files and orphans are deleted, and the
folders of other fingerprints go with it. Before: everything lost at exit. After: a restart finds
the tier where it left it.

## Related discussion

none

## Approach

- Each persistent entry gets a small index file beside its session file (`conv-N.meta`: identities,
  byte size, the live token/image ids, the checkpoint ids). A restart adopts an entry when its index
  reads back, its identity matches and the session file's size equals the recorded one; anything else
  in the folder is removed. No time-based cleanup: an entry goes only when the size cap needs room
  (least recently used first — a matched-and-read entry refreshes its use) or when a longer spill of
  the same conversation replaces it (an entry whose tokens are a prefix of the new one is superseded
  and deleted at write time). Reads in persistent mode leave the file in place; `close()` keeps it.
- The fingerprint itself had to be fixed first for this to work live: `--mtp` can name an rt
  directory (draft_vocab.bin only) whose drafter the Metal engine opens as the GGUF beside it, and
  the old fingerprint listed a pack layout that folder does not have — the fingerprint failed, which
  closed the tier. That fix is the first commit here and is open separately as #6305: **this PR
  depends on #6305** (rebase onto main once it lands).
- Compatibility: off by default; without `--conversation-spill-persist` the per-process folder
  behaves exactly as before. No format change to session files; the `.meta` index is new and only
  written in persistent mode. A damaged index drops just its entry, never the run.

## Validation

**Passed:**
- `g++ -std=c++20 -I engine/include <test>.cpp engine/src/core/conversation_file.cpp -pthread` + run,
  revision fc9110f06 (this branch), macOS arm64, Apple clang:
  - `conversation_spill_test`: 96 checks, 0 failed — includes the new persist block: a first run
    writes three entries into `fp-<hex>`; a second live engine of the model gets its own folder; a
    restart with the same fingerprint adopts all three and reads one back byte for byte; a read
    leaves the entry in place; at the cap (`--conversation-spill-mib`) the least recently used entry
    goes, not the oldest written; a grown conversation replaces its superseded prefix and entries
    that are no prefix stay; a restart with another config or another model fingerprint adopts
    nothing and removes the old folder; a cut index file drops only its entry.
  - `conversation_file_test`: 209 checks passed (includes the #6305 MTP-resolution cases).
  - `conversation_cache_test`: 13534 checks passed.
- Real-world proof: 74 conversations (85794 MiB) adopted after a restart.

**Failed:**
- none observed.

**Not run:**
- CUDA/HIP/SYCL builds and the full engine (`generate.cpp` wiring is host code compiled into the
  serve path; only the core tests above were built here). The Windows lock/rename paths of the
  stable folder were not exercised on Windows.

## Limits

- Adoption requires an identical model fingerprint: a moved-but-identical model still matches (roles,
  not paths), any byte change to a hashed file or a config change drops the tier's files at the next
  start by design.
- The cap applies across restarts: if `--conversation-spill-mib` is lowered between runs, adoption
  evicts least-recently-used entries down to the new cap.
- Two engines of the same model: only one holds the stable folder; the other spills into its own
  per-process folder (not shared, removed at exit).
