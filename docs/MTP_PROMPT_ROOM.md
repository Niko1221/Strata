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

**Measured on our Q8_0 4-way rig on 2026-10-16** (what it was measured on and every number is under
[Measuring it](#measuring-it)): the room did what this page says - it opened, the 675 MiB went into the ring, the block
came back before the first window and the decode ran unchanged - and it bought no time at the prompt sizes tried,
because the ring was not the limit there. It cost 3.4 to 4.0 s per roomed request on top of the read.

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
- Measured on the Q8_0 4-way rig only, at 3,080 and 14,859 tokens; on both the ring was not the limit, so the room
  cost its own rebuild and bought nothing. See [Measuring it](#measuring-it).

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

### Measured: Q8_0 4-way, 2026-10-16

**What it was measured on.** 2× RTX 3060 (11.6 GB) + 2× RTX 5060 (7.5 GB), layer split `auto` across all four cards,
Qwen3.8-Flash-Next Q8_0 pack, MTP runtime `rt`, `--spec 4`, `--kv-resident 65536`, `--max-context 262144`. The engine
was this branch's build (upstream main + the room), `--serve` only. The prompt chunk auto-picked 3,328 tokens with a
101-slot ring; the room's 135 slots take it to 236 while a roomed prompt runs. Each arm was a fresh engine, the
variable unset against `=0`, one request per size, `temperature 0`, 96/128 tokens out.

| prompt | arm | read (the engine's `prompt … read in` line) | of which the room | decode |
|---|---|---|---|---|
| 3,080 tokens | room | 20,647 ms (149.2 tok/s) | rebuild 2,461.6 ms + back 1,572.5 ms | 25.1 tok/s |
| 3,080 tokens | room (repeat) | 20,685 ms (148.9 tok/s) | rebuild 2,308.7 ms + back 1,580.9 ms | 25.9 tok/s |
| 3,080 tokens | standing | 16,171 ms (190.5 tok/s) | — | 26.4 tok/s |
| 14,859 tokens | room | 39,807 ms (373.3 tok/s) | rebuild 1,852.6 ms + back 1,566.1 ms | 31.2 tok/s |
| 14,859 tokens | standing | 35,794 ms (415.1 tok/s) | — | 32.7 tok/s |

The engine's `prompt … read in` line covers the room open and close (it is timed from the start of the request to
`room_close`), so what the read itself took is that minus the two room lines: about 16.6 s against 16.2 s at 3,080
tokens, and 36.4 s against 35.8 s at 14,859. **The ring bought nothing at either size** - it was not what the prompt
was waiting on - and the room's own 3.4-4.0 s is what the request paid.

Everything else behaved as designed. The start-up line read `a prompt of 0 tokens or more lends the drafter's 675 MiB
of experts on CUDA3 to the prompt path's ring (135 slots past its own)`. CUDA3's free VRAM while a roomed prompt ran
was 66 MiB, the same as standing (66 MiB): the freed 675 MiB was the ring. On the **first** roomed prompt of an engine
the block came back at another address (`came back at 0x744866000000, not 0x746dbc000000`) and the round/step graphs
were dropped and captured again - the decode after it ran at 25.9 tok/s all the same; every later prompt returned it
to the same address and nothing was dropped. No guard refused a `draft`, no request errored, and decode was unchanged
within noise (the two arms' expert-cache hit rate was within 1.3 points of each other).

This is the shape the reference measured too (90.9 s against 89.5 s on a 128K prompt): the room moves the ring, and on
a machine where the ring is not the limit, that is all it moves.

## Where this comes from

The shape is [mw00/project-maya#223](https://github.com/mw00/project-maya/pull/223) (`STRATA_GLM_ROOM_MIN`), which does
the same for its DFlash drafter's weights and its NextN block. What this port keeps is the mechanism: a decode-only
block leaves for the length of a prompt, and what takes it is the prompt path's ring. What differs is Strata's own
machinery - here the freed bytes are *spilled* past the expert cache's loan rather than added to it, because the loan
is one contiguous slice of the cache and cannot grow after the cache is sized at start.
