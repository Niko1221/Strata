# [Engine] fingerprint the MTP GGUF itself, so the SSD conversation tier stays on for GGUF drafters

Fixes #6281

## Problem and result
Since 0.2.16 the conversation cache has a default-on SSD tier. Before the first spill the engine fingerprints the model, so a spilled conversation is only ever restored by the model and configuration that wrote it. The fingerprint list assumed that `--mtp` always names an MTP pack folder: it asked for `<mtp>/dense.bin`, `dense.txt` and `experts.bin`. When `--mtp` names a single GGUF file (a drafter loaded from a GGUF, with at most a `draft_vocab.bin` next to it), those paths do not exist and the fingerprint fails at the first spill:

    strata serve: conversation cache: no SSD tier from now on: the model cannot be fingerprinted (session fingerprint: mtp <dir>/dense.bin: No such file or directory)

The tier then closes for the rest of the run. Disk sessions (`--session`) hit the same list.

After this change a GGUF drafter is fingerprinted by the GGUF file; the pack files are listed only when `--mtp` is a folder. The SSD tier stays on for GGUF-MTP setups.

## Related discussion
Issue #6281.

## Approach
- `session_model_inputs` now checks whether `--mtp` is a regular file: then the file itself is the drafter input (role `mtp gguf`); otherwise it keeps `<mtp>/dense.bin` under the unchanged role `mtp`.
- New `session_mtp_pack_files` returns the pack-only extras (`dense.txt`, `experts.bin`, optional `draft_vocab.bin`) for a folder and nothing for a GGUF; for a GGUF it adds `--mtp-draft-vocab` when given, since the engine loads that file too.
- The CUDA/HIP/Metal and SYCL `generate.cpp` call these instead of building the list inline.
- Nothing is relaxed: every file the engine loaded stays in the hash (size, head and tail samples) under its role, so another drafter, a different pack or a changed GGUF still yields another fingerprint and its spill is not restored. A missing required file still fails the fingerprint and closes the tier. For pack folders the list and roles are byte-for-byte as before, so existing session files stay valid.
- No placeholder files are needed or wanted.

## Validation
**Passed:**
- `g++ -std=c++20 -Iengine/include engine/src/core/conversation_file_test.cpp engine/src/core/conversation_file.cpp && ./a.out` on v0.2.16 plus this change (macOS arm64): 198 checks passed, including new ones: a GGUF drafter alone fingerprints without any pack file, editing that GGUF changes the fingerprint, a pack folder still lists dense.bin / dense.txt / experts.bin / draft_vocab.bin, and a pack folder without dense.bin still fails.
- `cmake -S engine/metal ... && cmake --build ... --target strata` (Metal, macOS arm64): see the build result noted in the PR thread; the changed `generate.cpp` is part of that target.

**Failed:**
none observed

**Not run:**
- CUDA, HIP and SYCL builds (the SYCL edit mirrors the CUDA one line for line).
- A live serve run with a GGUF drafter after the change: the spill-on log line (`conversation cache: SSD tier in ..., up to N MiB`, no "cannot be fingerprinted" line once a conversation is spilled) is the expected observation and is unverified here.

## Limits
- As before, an edit in the middle of a model file that keeps its size is not detected (head and tail are sampled).
