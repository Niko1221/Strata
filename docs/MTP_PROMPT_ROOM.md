# The drafter's experts are prompt workspace while a prompt is read (`STRATA_MTP_ROOM_MIN`, opt-in)

Two large VRAM users never work at the same time. The MTP draft layer's 512 routed experts (708 MB, 675 MiB) are read
only while tokens are **generated**: `record_rest` is their only reader. A prompt's passes - the draft layer's own K/V
(`record_front`, in both prefill captures) and the prompt path's `Prefill::draft_kv` - run the layer's dense tensors and
its K/V state, and never touch the 512 expert blobs.

With `STRATA_MTP_ROOM_MIN=TOKENS`, a prompt of at least TOKENS tokens (0: every prompt) takes their VRAM for as long as
it is read: the blobs are copied once into pinned host memory, the device block is freed, the prompt path is laid out
again with that much more expert ring, and the blobs are back - at the address the decode's captured graphs carry -
before the first generated token.

Unset (the default), nothing of this runs and the prompt path is laid out exactly as it was.

## Why

A prompt streams the experts a chunk does not hold in VRAM through the ring of staging slots, a copy of each blob is
issued `ring` slots ahead of its use, and the ring's size is the measure of how far ahead that is. The ring competes
with the expert cache for the same VRAM, and on a card where the cache took everything the reserve left, the ring is
what a decode-only 708 MB can buy.

The reference implementation ([mw00/project-maya#223](https://github.com/mw00/project-maya/pull/223), its own NextN
drafter on a GLM-5.3-Flash model) measured its own version on a 128K-token prompt: the ring went from a 160-slot cap to
439 slots (303 on the second card) and the prompt took 90.9 s against 89.5 s standing. That prompt was compute-bound;
what the loan moved there was the ring, not the clock. Expect a gain where the ring is the limit and nothing where it
is not.

**This port has not been measured on our hardware yet.** The port is offered with the mechanism and the lines to
measure it by, not with numbers: nobody has run a model with it on. What to run is under [Measuring it](#measuring-it).

## What it does

1. **At start**, `STRATA_MTP_ROOM_MIN` is read once. The room is `experts_bytes / max_blob` ring slots (135 on a
   Q8_0 pack, whose largest blob is 4.98 MiB, for 512 experts of 1,382,400 bytes each). The engine prints the token
   threshold, the MiB and the slots.
2. **Per request**, if the prompt has at least TOKENS tokens:
   1. The drafter's stream is synchronized, its streamed chain waited for, and the 512 blobs are copied into a pinned
      host buffer (taken once and kept: the room opens and closes around every long prompt, and a second read of the
      675 MiB `experts.bin` per prompt is what that avoids).
   2. The device block is freed, and every prompt path is laid out again - the one on the drafter's card with
      `Prefill::set_ring_room(room)`, so its ring is that many slots deeper. Those slots spill past the borrowed
      region (the loan itself does not move: `bytes_needed` does not count the room), in ONE `cudaMalloc` taken after
      the region's own buffers - a slot at a time, 135 mallocs of 4.98 MiB would round up to 2 MiB pages and want
      130 MiB more than the block that was freed. The spill makes the layout hybrid, and a hybrid layout cannot be
      relaid out - so this request keeps it, as `--prefill-head-alias` does.
   3. The prompt is read. The draft layer's K/V pass (`draft_kv`) runs as usual: it reads the dense tensors and the
      K/V state, not the experts.
   4. Before the first window: the paths are freed and laid out as they were, then the blobs are allocated again
      (cudaMalloc of the same size, then one DMA back). If the block comes back at a **different address**, the
      captured round/step graphs would read the old one, so they are dropped and captured again on the next round -
      the engine says so when it happens. `--prefill-head-alias` stops in that case instead (its verify graphs are
      all-or-nothing); here the graphs are per-round and cheap to re-capture. Nothing was seen to move on our boxes;
      the prompt path's spilled ring slots are what would move it.
3. **Guards.** While the experts are out, `draft`, `draft_first`, `chain_launch` and `record_rest` refuse with an
   error rather than launch a graph that reads freed memory. In `--serve` no decode runs between 2.1 and 2.4 anyway:
   the room is opened and closed inside one request's prompt.

The freed bytes reach the prompt path **only as ring slots**, because the ring is what `carve` lays out last: every
other buffer of a chunk is inside the region `bytes_needed` sized for it. A room therefore cannot enlarge a chunk
(that is `--prefill-chunk`'s or the auto scan's job, and it is decided at start, before any prompt runs).

## Costs

- 708 MB of pinned host memory (taken once, at the first roomed prompt).
- 708 MB over the PCIe link each way per roomed request, plus the prompt path's own rebuild.
- The roomed request's layout is fixed: its loan is the full chunk's, as with `--prefill-head-alias`, even if the
  prompt is short.
- Prompts below the threshold, every decode, and every other part of the engine are untouched.

## Limits

- `--serve` only, `--mtp` needed, `--batch` not supported (its slot drafters share the solo drafter's weights, so
  freeing them under the slots would be a use-after-free). The engine refuses each with the reason.
- A layer split is fine: the room goes to the prompt path on the drafter's card (`--mtp-first`: the first).
- `--pipeline-windows` is fine: the room closes before the request's first window.
- `--prefill-head-alias` is fine: when both apply, the head is unloaded and the prompt path is laid out ONCE, with the
  head's bytes and the drafter's together, and both come back before the first window (`room_close`).
- Not measured on our hardware; see below.

## Measuring it

The env var is the A/B switch, so two fresh engines with the same prompt do it (the style
[docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) asks for):

```sh
STRATA_MTP_ROOM_MIN=0  <engine and args>    # every prompt takes the room
<engine and args>                           # standing behaviour (the variable unset)
```

What to read:

- The startup line `STRATA_MTP_ROOM_MIN=...: a prompt of ... tokens or more lends the drafter's ... MiB ... (N slots
  past its own)` — N is what the ring gained.
- `strata serve: prompt room: ... tokens, the drafter's ... MiB are the prompt path's (rebuilt in X ms)` and
  `prompt room (prompt done): the drafter's experts are back in X ms` — the two costs per roomed request.
- The prompt's own time (the `prompt done` line and the `prefill` tokens/s the API reports) and, for the same
  requests, decode tokens/s (which should not move) and the free VRAM while decoding.
- `strata mtp draft timing` and the `round/step graphs are dropped` line: the address moved if that line appears, and
  the first decode after the prompt then pays the re-capture.

A prompt where the ring is not the limit (an agent's tool output of a few thousand tokens on a card whose cache holds
most experts) should show the costs and no gain; a long prompt on a card that streams nearly every expert is where the
ring has been worth its bytes before.

## Where this comes from

The shape is [mw00/project-maya#223](https://github.com/mw00/project-maya/pull/223) (`STRATA_GLM_ROOM_MIN`), which does
the same for its DFlash drafter's weights and its NextN block. What this port keeps is the mechanism: a decode-only
block leaves for the length of a prompt, and what takes it is the prompt path's ring. What differs is Strata's own
machinery - here the freed bytes are *spilled* past the expert cache's loan rather than added to it, because the loan
is one contiguous slice of the cache and cannot grow after the cache is sized at start.
