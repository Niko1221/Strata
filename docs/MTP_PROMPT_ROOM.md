# The drafter's experts are prompt workspace while a prompt is read (`STRATA_MTP_ROOM_MIN`, opt-in)

Two large VRAM users never work at the same time. The MTP draft layer's 512 routed experts (708 MB, 675 MiB) are read
only while tokens are **generated**: `record_rest` is their only reader. A prompt's passes - the draft layer's own K/V
(`record_front`, in both prefill captures) and the prompt path's `Prefill::draft_kv` - run the layer's dense tensors and
its K/V state, and never touch the 512 expert blobs.

With `STRATA_MTP_ROOM_MIN=TOKENS`, a request that **reads** at least TOKENS prompt tokens (0: every request) takes
those bytes for as long as it is read: the prompt path's ring is **pointed at the block itself**, and the decode's own
bytes are copied back into it before the first generated token. The block never leaves VRAM and never moves. TOKENS
counts the tokens this request reads, not the whole prompt: an agent turn that sends a 50,000-token conversation and
reuses 49,900 of it reads 100 tokens, and the room has nothing to buy there.

Unset (the default), nothing of this runs and the prompt path is laid out exactly as it was.

## Why

A prompt streams the experts a chunk does not hold in VRAM through the ring of staging slots, a copy of each blob is
issued `ring` slots ahead of its use, and the ring's size is the measure of how far ahead that is. The ring competes
with the expert cache for the same VRAM, and on a card where the cache took everything the reserve left, the ring is
what a decode-only 708 MB can buy.

The Maya checkout implemented the same room first for its DFlash drafter (local commit `7df369c`) and then for its MTP
config's NextN block (local commit `5ccf9a4`, whose expert slots leave the decode's pool while a prompt runs). Its
`7df369c` records the room as not yet measured and names the standing baseline on its byte-identical 100,535-token
read: 89.5 s (1,123 tok/s). Expect a gain where the ring is the limit and nothing where it is not.

**Measured on our Q8_0 4-way rig** (what it was measured on and every number is under
[Measuring it](#measuring-it)): the room did what this page says - it opened, the 675 MiB really went into the ring,
and the decode ran unchanged - and it bought no time at the prompt sizes tried, because the ring was not the limit
there. The numbers below are from its first shape, which laid the prompt path out again around the block and so cost
3.4 to 4.0 s per roomed request; the shape described here lends the block **in place** and has no rebuild at all.

**Reported on an 8 GB RTX 2060 SUPER by merbanan** (on the pull request, ported onto a 0.1.45-based branch; not run by
us): there the ring *was* the limit, and the room was worth **+17% prompt tokens/s at 4K, +17% at 16K and +23% at
32K**, with decode unchanged and the first-token logits bitwise equal to the unset run. The report is under
[Measuring it](#measuring-it).

## What it does

1. **At start**, `STRATA_MTP_ROOM_MIN` is read once. The room is `experts_bytes / max_blob` ring slots (135 on a Q8_0
   pack, whose largest blob is 4.98 MiB, for 512 experts of 1,382,400 bytes each). The engine prints the read-token
   threshold, the MiB, the slots and the card.
2. **The prompt path on the drafter's card is given those bytes** (`Prefill::set_ring_room_base`): slot `i` of the room
   is `experts + (i - rule) * max_blob`, so a slot IS a part of the drafter's block. Nothing is allocated, nothing
   spills and the room costs **no VRAM of its own** - the layout holds it for the engine's life, and the loan, the
   cache and `relayout` are untouched by it.
3. **Per request**, if the request reads at least TOKENS tokens (the prompt minus the prefix a cached conversation
   reuses):
   1. The drafter's stream is synchronized, its streamed chain waited for, and the 512 blobs are copied into a pinned
      host buffer (taken once and kept: the room opens and closes around every long prompt, and a second read of the
      675 MiB `experts.bin` per prompt is what that avoids).
   2. The room is switched on: the path's ring count goes from the rule's slots to the rule's + the room's
      (`set_ring_room_active`). **Nothing is laid out again** - the slots' own pointers do not move - so there is no
      rebuild, no `relayout` refusal and no loan to re-anchor.
   3. The prompt is read, staging its expert blobs over the drafter's bytes. The draft layer's K/V pass (`draft_kv`)
      runs as usual: it reads the dense tensors and the K/V state, not the experts.
   4. Before the first window: the path's streams are synchronized, the ring count goes back to the rule's, and the
      675 MiB are copied from the pinned buffer **back into the same block, at the same address**
      (`MtpDrafter::experts_reclaim`). The decode's captured round and step graphs carry that address, so nothing is
      dropped and no round pays a re-capture.
4. **Guards.** While the experts are lent, `draft`, `draft_first`, `chain_launch` and `record_rest` refuse with an
   error rather than read the prompt's staged bytes as experts. In `--serve` no decode runs between 3.1 and 3.4
   anyway: the room is opened and closed inside one request's prompt.

The block's bytes reach the prompt path **only as ring slots**, because the ring is what `carve` lays out last: every
other buffer of a chunk is inside the region `bytes_needed` sized for it. A room therefore cannot enlarge a chunk
(that is `--prefill-chunk`'s or the auto scan's job, and it is decided at start, before any prompt runs).

## Costs

- 708 MB of pinned host memory (taken once, at the first roomed prompt; on a host with 40 GiB already page-locked,
  merbanan reported the first `cudaHostAlloc` taking 24.4 s, which a later request does not pay again).
- 708 MB over the PCIe link each way per roomed request, and nothing else: no layout, no allocation, no graph.
- No VRAM of its own: the room is the drafter's bytes, which are already resident.
- Requests that read less than the threshold (an agent turn reusing most of its conversation), every decode, and every
  other part of the engine are untouched.

## Limits

- `--serve` only, `--mtp` needed, `--batch` not supported (its slot drafters share the solo drafter's weights, so the
  room would point one drafter's ring at another's experts). The engine refuses each with the reason.
- A layer split is fine: the room goes to the prompt path on the drafter's card (`--mtp-first`: the first).
- `--pipeline-windows` is fine: the room closes before the request's first window.
- `--prefill-head-alias` is fine: the head's rebuild lays the path out with the room already in it, and the two are put
  back independently (the room before the first window, the head with its own `head_restore`).
- Measured on the Q8_0 4-way rig, at 3,080 and 14,859 tokens, and on the shape that still rebuilt the prompt path;
  there the ring was not the limit, so the room bought nothing and its rebuild was what the request paid. Reported by
  merbanan on an 8 GB RTX 2060 SUPER, where the ring was the limit: +17% / +17% / +23% at 4K / 16K / 32K. The
  in-place shape and the threshold that counts the tokens a request reads were measured on the same Q8_0 4-way rig
  on 2026-10-19: a continuation that reads 7 tokens skips the room, and its first token came 0.19-0.21 s sooner
  (that is the room's lend and reclaim). See [Measuring it](#measuring-it).

## Measuring it

The env var is the A/B switch, so two fresh engines with the same prompt do it (the style
[docs/COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md) asks for):

```sh
STRATA_MTP_ROOM_MIN=0  <engine and args>    # every request takes the room
<engine and args>                           # standing behaviour (the variable unset)
```

What to read:

- The startup line `STRATA_MTP_ROOM_MIN=...: a request that reads ... prompt tokens or more points the prompt path's
  ring on CUDA... at the drafter's ... MiB of experts (N slots past its own)` — N is what the ring gained.
- `strata serve: prompt room: ... tokens to read, the prompt path's ring is the drafter's ... MiB (N slots past its
  own, lent in X ms)` — X is the snapshot's DMA, the whole cost of opening the room now that nothing is laid out again.
  The count is the request's, not the whole prompt: compare it with the `RESUME` line.
- `strata serve: prompt room (prompt done): the drafter's experts are the decode's again in X ms` — the second DMA.
- The prompt's own time (the `prompt done` line and the `prefill` tokens/s the API reports; the engine's `prompt ...
  read in` line covers the room's close too) and, for the same requests, decode tokens/s (which should not move) and
  the free VRAM while decoding.
- `strata mtp draft timing`, and that **no** `round/step graphs are dropped` line appears: the block cannot move.

An agent turn that reuses most of its prompt (say 50,000 tokens sent, 49,900 reused, 100 read) skips the room at
`STRATA_MTP_ROOM_MIN=1024`, which is what counting the tokens read is for. A request that reads a few thousand new
tokens on a card whose cache holds most experts should show the two DMAs and no gain; a long cold read on a card that
streams nearly every expert is where the ring has been worth its bytes.

### Measured: Q8_0 4-way, 2026-10-16

**What it was measured on.** 2× RTX 3060 (11.6 GB) + 2× RTX 5060 (7.5 GB), layer split `auto` across all four cards,
Qwen3.8-Flash-Next Q8_0 pack, MTP runtime `rt`, `--spec 4`, `--kv-resident 65536`, `--max-context 262144`. The engine
was this branch's build, `--serve` only. The prompt chunk auto-picked 3,328 tokens with a 101-slot ring; the room's
135 slots take it to 236 while a roomed prompt runs. Each arm was a fresh engine, the variable unset against `=0`, one
request per size, `temperature 0`, 96/128 tokens out.

**These numbers are the port's FIRST shape**, which laid every prompt path out again around the block and allocated its
bytes as a spill block of its own (2.3-2.5 s) before reading and again after (1.6 s). The shape this page now
describes lends the block in place, so those two rebuilds do not happen; the ring, the prompt and the decode are the
same either way, and the conclusion below is about the rig and the prompt sizes, not about the mechanism.

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
was waiting on - and the room's own 3.4-4.0 s was what the request paid in that shape.

Everything else behaved as designed. The start-up line read `a prompt of 0 tokens or more lends the drafter's 675 MiB
of experts on CUDA3 to the prompt path's ring (135 slots past its own), and they are back before the first window`.
CUDA3's free VRAM while a roomed prompt ran was 66 MiB, the same as standing (66 MiB): the freed 675 MiB was the ring.
On the **first** roomed prompt of an engine the block came back at another address (`came back at 0x744866000000, not
0x746dbc000000`) and the round/step graphs were dropped and captured again - the decode after it ran at 25.9 tok/s all
the same; every later prompt returned it to the same address and nothing was dropped. No guard refused a `draft`, no
request errored, and decode was unchanged within noise (the two arms' expert-cache hit rate was within 1.3 points of
each other). In the shape this page describes, that graph drop cannot happen: the block never moves.

This is the shape the Maya checkout implemented too: the room moves the ring, and on a machine where the ring is not
the limit, that is all it moves.

### Measured: RTX 2060 SUPER 8 GB, reported 2026-10-18

**What it was measured on.** Reported by **merbanan** on the pull request, on a port of this PR onto a 0.1.45-based
branch; the maintainers have not run this shape on this card. RTX 2060 SUPER 8 GB (sm_75), PCIe 3.0 x8, Ryzen 9
3900X, 92 GB RAM, Qwen3.8-Flash-Next IQ3_XXS, `--serve --mtp` (Q2_0 drafter, 675 MiB of experts),
`--resident-budget-gib 40` (fully page-locked), `--kv k8v4 --kv-resident 20480`, `--prefill auto`. The auto scan's
chunk was 6,656 tokens with a **96-slot ring**; the room adds **303 slots** (399 in all) and the chunk stays 6,656.

The ring was the limit on this card. `STRATA_PREFILL_RING` alone, paired A/B with 3 pairs against the 96-slot default
(prompt tok/s): 48 slots gave -6.3% / -3.4% / -4.6% at 4K / 16K / 32K; 192 slots gave +9.8% / +6.2% / +2.3% at the same
sizes, with decode -3.1 to -3.6 tok/s. Taking ring slots from the expert cache also shrinks the auto chunk
(6,656 -> 5,632) and costs decode after short prompts; the room gets the ring without either cost.

`STRATA_MTP_ROOM_MIN=0` against the same build unset (paired A/B, 3 pairs, prompt tok/s / decode tok/s):

| prompt | unset | room | paired difference (prompt) | decode |
|---|---|---|---|---|
| 4K | 448.4 | 527.2 | +75.8 (+17%) | -0.27 (noise) |
| 16K | 558.9 | 658.1 | +94.4 (+17%) | +1.42 (noise) |
| 32K | 584.0 | 719.7 | +135.7 (+23%) | -0.17 (noise) |

The lend cost 107 ms per roomed request. The first roomed request after a start paid 24.4 s once, for the 708 MB
`cudaHostAlloc` with 40 GiB already page-locked; later requests did not pay it. The first-token logits after a
15,521-token prompt were **bitwise equal** to the unset run (two runs per arm, all four identical), the 48-token
greedy replies matched, and the drafter's acceptance after the room was given back was normal (27 of 34), so
`experts_reclaim` restores the bytes intact. `STRATA_PREFILL_KEEP=64` (#2833) on top of the room changed nothing on
this card: the chunk stayed 6,656 and the result was +78.2 / +92.0 / +134.9.

What this report does not cover: HIP/SYCL, `--prefill-head-alias`, or a threshold other than `0`. The in-place shape
and the threshold that counts the tokens a request reads were measured on our Q8_0 4-way rig after it (below); `=0`
keeps every request roomed, as measured here and in the control there.

### Measured: Q8_0 4-way, 2026-10-19: the threshold that counts what a request reads

**What it was measured on.** 2x RTX 3060 (11.6 GB) + 2x RTX 5060 (7.5 GB), layer split `18,39,46`,
Qwen3.8-Flash-Next Q8_0 pack, MTP runtime `rt`, `--spec 4 --spec-min-p 0.5`, `--prefill auto:262144`,
`--kv fp16 --kv-resident 0`, `--max-context 262144 --vram-reserve-mib 700 --trim-stage-weights --vision`, 247 GB of
RAM. The config's other environment variables (`STRATA_PREFILL_MAX`, `STRATA_WARM_PROMPT`, `STRATA_MMQ_SLAB`,
`STRATA_PREFILL_STAGES`, `STRATA_PREFILL_KEEP`) belong to branches that are not in this one and were ignored; only
`STRATA_MTP_ROOM_MIN` is read here. The room was the drafter's 675 MiB on CUDA3 (135 slots past the ring's own).

Two engines from the same branch, one start each: the parent commit (whole-prompt test) against the fixed one
(tokens-read test), `MIN=1024` in both. The same client sent one 5,042-token chat (temperature 0, 32 tokens out,
roomed in both arms) and then three continuations: the assistant's answer echoed back plus a one-line user turn,
3 tokens out. A third start of the fixed binary at `MIN=0` is the control.

| request | parent: whole prompt | fixed: tokens read |
|---|---|---|
| cold: 5,042 read | room 5042 tok; lend 603 + close 104 ms; read 17,454 ms; first token 17.62 s | room 5042 tok; lend 534 + close 107 ms; read 17,323 ms; first token 17.47 s |
| turn 2: 5,035 reused + **61 read** | room **5096 tok**; 102 + 104 ms; read 2,054 ms; first token 2.110 s | **no room**; read 1,813 ms; first token **1.869 s** |
| turn 3: 5,089 reused + **7 read** | room **5096 tok**; 102 + 104 ms; read 360 ms; first token 0.409 s | **no room**; read 153 ms; first token **0.203 s** |
| turn 4: 5,089 reused + **7 read** | room **5096 tok**; 102 + 105 ms; read 344 ms; first token 0.390 s | **no room**; read 152 ms; first token **0.199 s** |

The parent lent the room for the whole 5,096-token prompt on every continuation while the request read 61, 7 and 7
tokens; its lend and reclaim (the two numbers in each cell) are the whole difference. Turns 3 and 4 are the clean
comparison - turn 2 also restores the 5,035-token conversation, so its read is not the room's cost alone: the room's
~206-207 ms is gone, and the first token comes 206 ms and 191 ms sooner. The cold request still takes the room in
both arms; its 131 ms difference is run-to-run noise (the control start read the same prompt in 16,286 ms).

The `MIN=0` control opened the room on the same continuations with `23 tokens to read` and `7 tokens to read`
(lend ~103 ms, close 105-106 ms): first token 0.432 s and 0.415 s on turns 3-4, the parent's ~0.4 s again. So `=0`
is unchanged, and the savings above are the room being skipped, not the new binary.

Every continuation answered `done.` in all three arms. The cold answer (temperature 0) matched between the parent
and fixed arms but differed in the control run after the same opening, so run-to-run bitwise determinism is not
claimed here. What this measurement does not cover: more than one run per timing, HIP/SYCL, `--prefill-head-alias`,
or a threshold other than 0/1024.

## Where this comes from

From the Maya checkout (`project-maya`): local commit `7df369c` (`STRATA_GLM_ROOM_MIN`) does this for its DFlash
drafter's weights and its prestage ring, and `5ccf9a4` does it for the MTP config's NextN block, whose expert slots
leave the decode's pool while a prompt runs; `d4d0792` rounds the tail's slots up so the reload lives on the bytes the
shrink hands back. What this port keeps is the mechanism: a decode-only block becomes the prompt path's ring for the
length of a prompt, and what it costs is the block's own contents. What differs is Strata's own machinery - here the
room's slots are *carved from the drafter's own block* rather than from freed cache, because Strata's drafter owns its
512 experts as one contiguous allocation and never shares them with the expert cache; lending them in place is the
Maya checkout's later shape (`5ccf9a4`), and it is what removes the rebuild this port started with.
