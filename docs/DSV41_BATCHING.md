# Batched decoding for DeepSeek-V4.1 (`strata-dsv41`)

Status: **design, partly built**.  The scheduler's CPU half (M1), the chunked-decoding path in the server (Path B), the
engine's batch protocol (BGEN/BADM/BT/BDONE/BSTOP) and N slots that take turns on the one model by parking their states
(`--batch N`, `--slot-burst K`) are implemented and tested; the kernels (M2 varlen attention, M3 batched MoE) are not -
they are what turns the present interleaving of requests into real throughput.  Scoped for the `deepseek41` engine and
the HTTP server that drives it. Numbers are from a 4 x NVIDIA L20 box (184 GB VRAM, PCIe 4.0, no NVLink) running the IST-DAS
2.25 bpw pack (274 GB, 1M context).

## M2, re-scoped: the attention kernel already takes several queries (2026-10-26)

The design assumed M2 needed a new "varlen attention" kernel.  Reading the code it does not: `AttnRows` already describes
**T queries**, each with its own window offset and counts (`ds4_kernels.cuh:154-160`: `win0[t]`, `nwin[t]`, `nsel[t]`, and
`sparse_attn(..., int T, ...)` takes the selection table with a per-query stride), and
`ds4_attn_split.cuh` runs them in one launch (`const int t = blockIdx.z; ... ar.nwin[t] ... ar.nsel[t]`).  That is how
DSpark's verify (T = 2..5) already works.  The queries' window rows come from one buffer at a per-query offset, and their
compressed rows are indexed per query into one compressed-KV buffer.

So M2 is not a new kernel: it is **per-sequence KV regions and a per-step table**, i.e.:

* each sequence's window ring at `seq * kWin` rows of the one window buffer (the ring arithmetic keeps the sequence's own
  position, so `pos % kWin` becomes `seq * kWin + pos % kWin`);
* each source's compressed KV and index keys in disjoint row ranges of their buffers (`seq * cap` rows), the compressor
  kernels writing at `base + pos / ratio`, the indexer selecting within the sequence's range;
* a per-step `AttnRows`: `T` = the sequences decoding this step, `win0[t]`/`nwin[t]`/`nsel[t]` from each sequence's state,
  and the selection table built per sequence;
* the model's `pos` (and `toks`, the checkpoints) per sequence instead of one - the `Run` in the serve loop already holds
  that half.

The cost that decides the design is **memory**: the compressed KV is ~2.5 GB at a 1M-token context, so the per-slot KV has
to be capped (e.g. 32K tokens a slot: a few hundred MB for 2-3 slots) - the batch's context budget, not the whole
`--max-context`.  With that, M2 is bookkeeping in the model around buffers the kernels already address, and the risk is
the one the design names: a row counted wrong does not crash, it answers subtly wrong - hence a parity test against the
present per-query calls is the first thing to write.

## M2 in the model: the sites to change

The kernel side is ready (above).  In `ds4_model.cu` the per-sequence base threads through these places, in this order:

1. **the window ring's allocation** (`dalloc(d, l.win, g.win * win_rb(), ...)`, ~3748): `nseq * g.win` rows, one ring a
   sequence; every use below takes `l.win + seq * win_rb()` (the ring functions already address rows from the pointer they
   are given, so nothing in the kernels changes);
2. **the decode path's ring traffic**: `ext_to_ring` (~6914), `ring_to_ext` (~6969), `ring_restore` (~7196) and the
   prompt path's `ring_to_ext` (~7283) - each call gets the sequence's ring;
3. **the checkpoint and slot code** (~4913, ~4976, ~5058): a window checkpoint copies `g.win * win_rb()` bytes of *one*
   ring - with N sequences it is N rings (or the sequence's, once checkpoints are per sequence too);
4. **the compressed rows**: `compress_win` writes at `p0 / ratio` and `index_score`/`sparse_attn` select rows - all of
   them need the source's `kv_base` (a sequence's rows are `kv_base .. kv_base + cap`), which is what
   `ds4_batch_attn.hpp` puts into the step's selection table;
5. **the model's state**: `pos`, `toks`, `ctoks` and the checkpoints become per sequence (an array of `nseq`), and
   `feed`/`step`/`sample` take the sequence they work on - this is the `Run` the serve loop already keeps, moved into the
   model.

Sites found for the compressed rows (the next step): `compress_win` is called at ~7235, ~7346 and ~9135 with the layer's
`l.c_kv`/`l.c_score` and a latent scratch (`c.latent`, `d.latent`), and the finished group's row is written into the
source's `ckv`/`ik` afterwards - that write (and the indexer's reads, `index_score`) is where a sequence's `kv_base` goes,
not the compress call itself.

**The write half is in (2026-10-26).**  A finished group's row is written at `kv_base + pos / ratio` now, where
`kv_base` is `Model::Impl::kv_base(src)` = `vseq * src.cap` and `vseq` is the sequence the pass serves (0 until the
batched step sets it): both paths' `finish` calls (`fi.pdst` for the index keys, `fc.pdst` for the compressed KV), the
copies that carry a source's new rows to the other devices' copies (`replicate`'s offset, the group-to-group `x.dst`),
and the tables themselves are allocated for `nseq` regions of `cap` rows, like the window rings.  With one sequence the
base is 0 and every address is the `g_first` rows it always was; `ds4_batch_attn.hpp`'s `write_row(layout, pos, ratio)`
is the same address as a function of the layout, tested without a GPU.  What still starts at row 0 is the *read* side -
`index_score`'s scan and the selection rows `sparse_attn` takes - which is the step table's work: a second sequence's
rows are written in their own region but not yet read from it.

**The state is per sequence (2026-10-26).**  `pos`, `toks`, `ctoks`, the four checkpoints and `slot_held` are one
`State` now, `nseq` of them (`seqs`), and a pass works on `seqs[vseq]` - `state()`: the chunked prompt and the window
path, `rollback`, `checkpoint`/`restore`/`pin`/`check_positions`, `slot_save`/`slot_load`, `park_save`/`park_load`
and the Engram's token rows all go through it, and `Model::pos()`/`tokens()` read it.  The compressors' partial group
is per sequence as well (`comp_kv`/`comp_sc`: `nseq` of `ratio x hd` floats a layer, its own at `vseq`), so a
sequence's half-finished group belongs to it alone - the checkpoint, slot and park paths copy the pass's.  With one
sequence - `vseq` is 0 everywhere today - every access is the member it was.  What it unlocks is the batched step switching a pass to another
sequence's state without parking it in host memory; what it still needs is the read side of the compressed rows (the
step table above), and `feed`/`step`/`sample` naming the sequence they work on instead of today's default.

**The regions are read from, and a sequence on its own answers what sequence 0 does (2026-10-26).**  The indexer scans
the pass's sequence's region now (`ik_rows`: the table at `kv_base` rows) and its picks are made absolute for the attention
(`rows_add_i` adds that base, so `sparse_attn` reads the one table from its base - the form a single step over several
sequences needs), and the checkpoint / slot / park paths take the pass's ring and region too (`win_of(l)` is the pass's
ring, `ckv_rows`/`ik_rows` its table) - with one sequence every pointer is the buffer's base, as it was.  `STRATA_DS4_VSEQ=n` runs a pass on sequence n (which brings its
own region along: `nseq >= n + 1`), and `tools/dsv41/batch_seq_parity.py` (the window
`tools/dsv41/windows/batch_seq_window.sh`) runs the same prompts on sequence 0 and sequence 1 in two loads - a 40-token
prompt (12 tokens), a 300-token one (140 tokens: past the 128-token ring, across many compression groups) and a prompt
sharing a 20-token prefix with the first (the model's own checkpoint/restore) - and sequence 1 writes exactly what
sequence 0 does.  That is the region arithmetic checked end to end, which is what a row counted wrong would break.

**A slot's conversation stays in its sequence (2026-10-26).**  The serve loop parked a slot's conversation between its
bursts because the model held one; with the state, the rings, the compressed rows and the compressors' group per sequence
it does not: `STRATA_DS4_BATCH_LIVE=1` gives each slot a sequence of its own, the pass switching with `Model::use_seq`
instead of `slot_save`/`slot_load` (or the cards' `park_save`/`park_load`), a solo GEN takes the sequence after the
slots', and `reset()` clears the pass's sequence only.  The probe with two slots and the three-client HTTP check pass with
the env on, and the slots take turns more finely for it (14 handovers over 47 tokens against the parked path's 8, no
switch cost at all) - what is left of M2 is the single-launch step, not the switching.

### The single-launch step: what is in, and what it needs on this box (2026-10-27)

The model side is done: every sequence's ring, compressed rows, compressors' group and state live in their own region
(`nseq`), a pass can be pointed at any of them (`use_seq`), the indexer's picks are absolute rows of the one table
(`rows_add_i`), and `Finish` takes a position a row (`pos_rows`).  The parity check proves the regions end to end.

What one launch over T sequences still needs, in the order it has to be built:

1. **a per-row descriptor through `layer_chunk`** (`pos[t]`, `seq[t]`): the window rows side by side at
   `t * (win + 1)` of the chunk scratch, `finish`'s `pos_rows` (in), `ext_to_ring` per row, `compress_win` and the
   ik/ckv writes per row (each row's own group and `kv_base`), the indexer per row - the attention is already one call
   (`AttnRows` carries `win0`/`nwin`/`nsel` per query and the picks are absolute);
2. **the cross-group rows**: with `--tp 2` on four GPUs half the sources are read by the other group's layers, and the
   new rows travel today through the staged host-mapped hand-off (one block a chunk, a flag and acks a window).  For T
   rows at T different offsets that becomes: one copy of the step's T latent rows to an owning group (`T x hd` floats),
   then each reader's own `finish` into its copy - the pack's quantization is deterministic, so the copies stay
   bit-identical - about 32 KB a layer a step, ~8 MB a step in all, which PCIe pays in ~0.3 ms.  **Or the step runs in one
   tensor-parallel group** (`--tp 4` on this box), where no source is remote and the hand-off is not needed at all: measured
   on 2026-10-27, the probe with two slots (every check) and three clients over HTTP (100 tokens each, 5.26 s of wall, 52
   handovers) are green at `--tp 4`, 8K context.  That is a route to land the step first and measure what one group costs;
3. **the serve loop**: one step over every active slot (the live mode is in, so nothing is parked), then a `BT` a slot.
   Two ways into the step, and the second is the shorter one now that the chunk path is per-row: (a) `forward_rows` over
   `layer()` - the window path, which needs the per-row descriptor in `layer()` and its plan (`plan_window` takes the rows
   now) - and (b) **one chunk of T rows through the chunk machinery** (`run_dev_chunks` with `nk = 1`), since `chunk_rows`
   is already the per-row path the in-process A/B proves: the entry fills `h_emb` a row (each sequence's token) and the
   Engram rows a row, and `Run` carries the descriptor to `run_layer`; what it still needs is that descriptor on `Run` and
   each row's logits sampled from the chunk's output rows (the head is per row already);
4. **the measurement**: `route_overlap` bounds what the shared expert reads can save at 1.15-1.28x, so the step's wall
   at sixteen slots should land below the stock engine's 27.35 s (the one-pass-per-slot live mode is 31.08 s).

Until (2) is built the engine keeps one pass a sequence: the live mode buys the waiting, the single launch the
throughput.

**The per-row chunk path is in and validated in process (2026-10-27).**  `StepRows` - a position and a sequence a row -
goes through `layer_chunk`: the window block a row (`[the ring's rows][the token]` at `t * (win + 1)`), the compressor
and the ik/ckv writes a sequence's (its own group, its own completed group), the indexer a row (its own scan, its picks
made absolute), `AttnRows` in its explicit form (one launch).  The window rule a row is **`win - 1` ring rows and the
token's** (or `p` and it when the position is younger than the ring), and the chunk scratch holds `t * (win + 1)` rows,
not `win + C` - both found by getting it wrong first.

The check that says so is **in process** (`STRATA_DS4_STEPROWS=2`): the same chunk through both paths on the same state -
the layer's rows, its ring, the compressors' group and the table rows the chunk writes snapshotted and restored between
the runs - with the rows and the written bytes compared directly.  A black-box A/B cannot do it: at `--prefill-chunk 8`
two *identical* contiguous runs differ (tokens 3 to 6), at `--tp 4` too, so the engine is not deterministic in that shape
(the same prompt at the default chunk sizes is byte-identical run to run: the three-client check has compared it with
production all along).  On a 300-token prompt read in chunks of 8 - 38 chunks through every layer - the harness reports
**no difference**: the rows, the compressed KV and the index keys are bit-identical, and both paths write the same 12
greedy tokens.

The order above is also the safe order to build it: (1) with `nseq = 1` changes nothing and can be checked by the
probe's GEN/BGEN equality; (2) with two sequences' rings side by side but only one active is still the present behaviour;
(4) is where a second sequence's rows start to be read, and where the parity test (`ds4_attn_batch_test`) becomes the
reference.  Items 1-5 are in, and a sequence other than 0 answers on its own: what is left of M2 is the **single-launch
step over several sequences** - per-row positions (`finish` takes one `pos` a window today) and the step table's rows,
`ds4_batch_attn.hpp`'s `rows`/`place` - and `feed`/`step`/`sample` naming the sequence.

## What a batched MoE can expect (measured, 2026-10-26)

M3 assumes the expert weights are read once for the tokens of a step, which pays only when those tokens route to the same
experts.  The engine writes its routing when asked (`DS4_TRACE=file`: a record a route request - u8 layer, u8 T, then
T x k_exp int16 expert ids), and `tools/dsv41/route_overlap.py` measures it.  Two sequences decoding at the same time
(`--batch 2 --slot-burst 1`, two unrelated prompts) share **25.5 %** of their six experts (1.53 of 6); consecutive tokens
of one sequence share 22.1 %.

Batching two tokens therefore reads 10.5 expert rows instead of 12 (**1.15x**), three tokens 14.9 instead of 18
(**1.20x**) - not the 2-3x the design assumed.  Pairing the requests by a shared prompt helps only a little: two requests
with a 1.3K-token system prompt in common share **32.7 %** of their experts (1.96 of 6), so 2 tokens still read 10.0 rows
instead of 12 (**1.20x**) and 3 read 14.1 instead of 18 (**1.28x**).  A batched MoE still saves the launches and a little dispatch, and a
workload whose prompts are much closer (one agent template, short answers) could route together better, but on this box
with these prompts the read side - where the time is - hardly moves.  Raising the overlap itself (grouping requests that
share a prompt, or a scheduler that pairs them) is a cheaper lever than the kernel, and worth measuring first.

## Concurrency at the production context (measured, 2026-10-27)

The box's production configuration is 1M tokens of context; these are real runs in it, three to sixteen clients at once,
each request asking `max_tokens: 262144` (256K) and leaving after it has read its share (thinking off, so the tokens
counted are the answer's).  The stock engine is the one without `--batch`; "this branch" is `--batch N` with each slot's
state in its own sequence (M2).

| clients | engine | wall | first token per client | handovers | aggregate |
|---:|---|---:|---|---:|---:|
| 3 | stock (no `--batch`, the server's FIFO) | 14.55 s | 0.50 / 5.81 / 10.84 s | 2 / 900 | 61.9 tok/s |
| 3 | this branch, `--batch 3` | 14.43 s | 0.31 / 0.76 / 1.04 s | 167 / 900 | 62.4 tok/s |
| 6 | `--batch 6` | 20.46 s | within 1.44 s | 181 / 1200 | 58.7 tok/s |
| 8 | `--batch 8` | 14.74 s | within 1.96 s | 119 / 800 | 54.3 tok/s |
| 10 | `--batch 10` | 18.50 s | within 2.50 s | 133 / 1000 | 54.1 tok/s |
| 12 | `--batch 12` | 22.12 s | within 2.95 s | 167 / 1200 | 54.3 tok/s |
| 16 | stock (no `--batch`) | 27.35 s | within 25.28 s (last: 25.93) | 15 / 1600 | 58.5 tok/s |
| 16 | `--batch 16` | 31.08 s | within 4.15 s | 219 / 1600 | 51.5 tok/s |

The two things this says:

* **Concurrency is latency and fairness, not throughput.**  The stock engine runs the three answers one after the other -
  the third client's first token waits for the first client's whole answer (10.84 s) - while the branch admits all three
  at once (first tokens within 0.74 s) and interleaves them (167 handovers against 2).  The aggregate is the same (61.9
  against 62.4 tok/s): each slot still runs its own pass, so the weights are read once a sequence; the single-launch step
  (M2's remainder) and the grouped MoE (M3) are what would move the aggregate, and the routing measurements above bound
  that at 1.15-1.28x on this box.
* **Sixteen concurrent requests fit at 1M context** (17 sequences: the slots and the solo one).  The price is expert
  residency: a sequence's compressed KV is ~0.54 GB a GPU (~2.2 GB in all) taken from the experts, so the cards hold 4224
  of 4224 experts with one sequence, 3662 at 13 sequences and 3441 at 17 - and the aggregate falls from ~62 to ~51 tok/s
  as the displaced experts are read from RAM.  The first-token latency stays within 0.7-4.2 s at every count.  Capping
  what one slot may hold (the design's 32K a slot, a few hundred MB) is the lever that would raise both.
* **The same 16 slots at a 32K context keep every expert in VRAM** (4224 of 4224, the tables 0.30 GB against 9.14 at
  1M) and score 57.2 tok/s aggregate on the same workload - so of the 62 -> 51 tok/s, the expert displacement is about
  half and the other half is what sixteen sequences' passes and cache working sets cost by themselves.
* **At sixteen clients the stock engine's wall is not much worse** (27.35 s against 31.08) because it runs them one at
  a time at full speed (58.5 tok/s); what it cannot do is start them together - its last client's first token comes at
  25.93 s against 4.15 s at worst.  So the branch's honest claim at this size is *waiting*, and the aggregate only comes
  with the single-launch step.

### Where the sixteen-slot run's extra 3.7 s goes (measured, 2026-10-27)

The engine says it itself at load: with one sequence "the experts VRAM leaves over (~0.0 GB) are read in place from the
model's files"; with the sixteen slots' seventeen sequences "~13.8 GB".  The 1M-context compressed KV is 9.14 GB a GPU
for them against 0.54 GB for one, and the experts it displaces are read per token.  The 32K control (all 4224 experts
resident, the tables 0.30 GB) takes 28.0 s for the same sixteen clients against the stock engine's 27.35 s: **+2.4 % is
the interleaving itself, ~+11 % is the spill**.  Three knobs were measured against the 31.08 s baseline on the same
workload:

| 16 slots, 1M context | wall | aggregate | note |
|---|---:|---:|---|
| stock engine (one at a time) | 27.35 s | 58.5 tok/s | no concurrency |
| `--batch 16` (baseline) | 31.08 s | 51.5 tok/s | 13.8 GB of experts outside VRAM |
| `--slot-burst 32` | 31.14 s | 51.4 tok/s | no gain, and the first token worsens (9.45 s against 4.15) |
| `STRATA_DS4_BATCH_NOSPEC=1` (drafter off) | 36.52 s | 43.8 tok/s | the drafter is worth 5.4 s even here - keep it on |
| `--experts 5` | **28.67 s** | **55.8 tok/s** | 17 % less expert traffic a token recovers nearly all of it |

So the levers, in the order they pay: the **single-launch step** (amortise the weights over the sequences, M2's
remainder), a **per-slot context cap** (the tables are what displace the experts - the 32K control is 2.4 % from the
stock engine), and `--experts 5` when throughput matters more than the last expert.  Warming does not help the steady
state - the deficit is permanent VRAM, not a cold cache - but it does help the *first* tokens, which pay the pack's cold
reads and the empty RAM store after a load.

### The other model family: Qwen3.8-Flash-Next (measured, 2026-10-27)

The same questions on the Qwen path (`strata`, the ISTA-DASLab IQ2_XS pack, 69 GB on the same box), which already decodes
several requests in one window (`"parallel": N`, `--batch N`): nothing in this branch touches it, and the measurements
say both that it runs normally on this tree and what its own batching is worth as a reference.

| context | parallel | clients | max_tokens asked | wall | first tokens within | handovers | aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32K, `--kv int8` | 1 | 3 | 256K | 12.02 s | 8.24 s | 2 / 300 | 25.0 tok/s (cold cache) |
| 32K, `--kv int8` | 4 | 4 | 256K | 8.77 s | 1.72 s | 377 / 400 | 45.6 tok/s (1.8x) |
| **256K, `--kv int8`** | **1** | **3** | **128K** | **4.11 s** | **2.87 s** | **2 / 300** | **73.0 tok/s** |
| **256K, `--kv int8`** | **4** | **4** | **128K** | **7.78 s** | **1.87 s** | **364 / 400** | **51.4 tok/s** (cache 96%) |
| **256K, `--kv int8`** | **8** | **8** | **128K** | **20.69 s** | **6.11 s** | **769 / 800** | **38.7 tok/s** (cache 84%) |
| 1M, `--kv q4_0` | 1 | 3 | 256K | 5.99 s | 3.99 s | 2 / 300 | 50.1 tok/s |
| 1M, `--kv q4_0` | 4 | 4 | 256K | 12.92 s | 8.96 s | 269 / 400 | 31.0 tok/s (cache 68-78%) |

Qwen's own batching at a chat context is a real aggregate win (45.6 against 25.0 tok/s at four requests, and 377
handovers against 2) where the dsv41 slots are still one pass each - that is what M2's single-launch step and M3 would
buy, and the reference this branch is measured against.  At the 256K context with 128K asked, four requests are the
sweet spot (51.4 tok/s, first tokens within 1.87 s) and eight cost (38.7 tok/s, 6.11 s), while one at a time keeps the
highest aggregate (73.0 tok/s) at the price of the queue - the expert cache goes from 96 % at four slots to 84 % at
eight, which is where the aggregate goes.  At 1M context it needs the KV at `q4_0` (with `--kv int8`, 45 KB
a token, the 1M-token KV leaves no VRAM for its expert tier and the engine refuses), and there the four slots *cost*
aggregate (31.0 against 50.1 tok/s) because the KV leaves the expert cache at a 68-78 % hit rate - the same trade this
branch's 1M-context numbers show.  The first case after a load pays the pack's cold reads through the file cache (the 32K
parallel-1 rates above), which is why the single-request numbers of the first case are lower than the later ones.

### The batch's geometry, computed (M5, measured 2026-10-27)

The engine reports what a batch costs on the model and the cards it was loaded on, from the pack's own geometry and the
cards' measured memory - nothing in it is a constant of a model or a box.  A sequence's tables are the sources' `cap` rows
of their two rows' bytes (`win`, the head dims and the ratios come from the pack); the room is `cudaMemGetInfo` after
the load; the count is what half that room allows on top of the sequences in place; the experts are what the cards
actually hold.  Same build, this box, `--batch 16`:

| context | a sequence's tables | room after the load | more slots at half the room | experts in VRAM |
|---|---:|---:|---:|---:|
| 8K | 8 MB | 17.4 GB | +1154 (a memory bound only) | 15360 / 15360 |
| 1M | 965 MB | 9.75 GB | +5 (22 in all) | 13190 / 15360 |

The 1M row is the policy the sixteen-slot runs asked for: each sequence's tables cost ~1 GB, the seventeen in place cost
~16 GB of the experts' room, and only about five more fit before the cards are full - so a slot count should be chosen
from this number (and from what the switch costs, which is the burst's tuning), not from a flag written by hand.  The
count is a memory bound: the protocol's slots and the scheduler bound it too, and at a chat context the memory is not what
limits it.

### The slot's context, measured and carried to the next load (M5, 2026-10-27)

`--max-context` sizes *every* sequence's tables, so how many a box holds is decided by a number in the config.  The engine
closes the loop: at the end of a load it computes the context at which the sequences in place would leave the experts the
room they have (`per_seq`, the free room and `nseq` are all measured), writes it beside the pack
(`batch-plan.txt`), and with `STRATA_DS4_AUTO=1` the next load sizes a slot's tables from it - it only shrinks, so the
chain converges instead of oscillating.  This box, seventeen sequences:

| load | a slot's context | tables a sequence | room left | experts in VRAM |
|---|---:|---:|---:|---:|
| first, at the pack's 1M | 1048576 | 964 MB | 9.75 GB | 13190 / 15360 (86 %) |
| second, the plan applied | 373521 | 344 MB | 11.9 GB | 14809 / 15360 (96 %) |

The same seventeen slots keep almost every expert once each may hold 373K tokens instead of the pack's million - the
number the sixteen-slot runs asked for, measured and applied rather than picked by hand, with the count that room allows
(`batch_fit_half`) rising from 22 to 34 with it.

### The slot's burst, measured (M5, 2026-10-27)

`--slot-burst K` - how many tokens a slot decodes before another gets the model - was a number in every script (8, 16,
32).  The engine measures it now: the turn's switch against the burst's own time, the shortest burst whose switch is at
most 5 % of it.  On the live path there is nothing to copy, the switch measures 0.0 ms, and it picks **one token** - the
three-client run then interleaves 158 times over 300 tokens against 52 with `--slot-burst 8`, at the same wall (5.27
against 5.26 s); with parked states the measured ~4 ms switch against ~15 ms a token gives six.  Neither the model nor the
number of cards enters the rule, and the trace prints what it picked (`ds4 slot: ... burst 1`).

### M6: the monitor's parallel view (2026-10-27)

The built-in API monitor (`/api-monitor`, `serve/web/monitor.*`) shows the batch as it runs, in the page's own vocabulary -
the same cards, panels and `--st-*` tokens: a row of slot cells that light while a slot holds a request, an area chart of
the requests in flight over four minutes (with the peak marked), and four facts (in flight, slots busy, waiting, peak).
The panel draws itself only where the server serves more than one request at a time, so a single-request deployment looks
exactly as it did.

The data is the engine's own: `/v1/status`'s `concurrency` gained `slots` (a slot's busy flag), `busy` and `waiting`,
read from the engine's slot bookkeeping.  Measured on the box (two slots, three clients, 8K): `slots=BB` while both slots
held requests, `B.` as one finished, `..` after - and the three clients answered in 5.16 s with 164 handovers.  The page
serves with `--api-monitor` and the panel's seven elements are in the served HTML.

What is *not* in it yet: the requests list does not name the slot a request holds (the engine knows it inside the batched
path), and the queue figure falls back to "in flight minus busy" when the engine reports nothing.

## Where a token's time goes (measured, 2026-10-26)

Decode on this box costs ~12.5 ms a token and the context barely moves it: 79 tok/s at a 216-token context, 90 tok/s at
3.5K, 90 tok/s at 10K, 77 tok/s at 20K (greedy, one request at a time, the batched engine; prompt reads 585-590 tok/s from
3.5K on).  The split-KV attention is documented in its own header at ~125 us a query (`ds4_attn_split.cuh`), ~1% of the
step, and even at 20K the whole attention path is a few percent.  So **M2 buys overlap, not throughput**: the time is in
the MoE (M3), which is where the 2-3x is.  M2 stays the prerequisite for a batched step - each token's attention needs
its own KV and positions - not the speedup itself.

## Where it stands

`strata-dsv41 --serve` reads one request at a time: `GEN <max_new> [keys] <ids>`, `GENI` for pictures,
`STOP`, `QUIT`. The model holds **one** conversation (its tokens, its KV rings per layer, the compressors'
partial groups, the index keys); `--serve`'s slots are conversations *saved to disk and restored*, i.e. a
context switch, not parallel decoding. The server queues requests behind one FIFO.

Measured on the box above:

| work | time |
|---|---:|
| one 300-token answer | ~5 s (65 tok/s) |
| three 150-token answers, sent together | ~6.5-21 s, i.e. the sum of the three |
| 6.6K-token cold prompt | ~9.6 s (654-690 tok/s prefill) |

Aggregate throughput therefore stays at roughly one request's worth, however many clients wait: the third
user is not slower *per token*, they simply start later.

## What batching would give

A step that decodes one token for N sequences does the same weight reads and the same MoE gather for N
rows instead of one. On this box the expected shape is:

* three concurrent answers finish in roughly the time of one (+20-40 %), instead of 3x;
* per-sequence decode drops from ~65 to ~45-55 tok/s while the aggregate rises to 135-165 tok/s.

It does not make a single request faster; it makes the third user wait for the first instead of for the
first, second and third.

## Design

### 1. Sequence objects

`struct SeqState { std::vector<int32_t> tokens; int32_t pos; KvBlocks kv; Sampler sampler; ImageSpans images;
StopReason stop; }` — everything the current `Model` keeps for its one conversation moves into one of
these. The weights stay shared and resident (they are the bulk: 149 GB of experts + ~16 GB dense).

### 2. KV cache

Today the KV lives in per-layer rings sized by `--max-context` (`kv: fp4`, `kv_resident`). Batching needs
per-sequence rings and a block table, i.e. paged attention: blocks of, say, 256 tokens, a free list, and a
per-sequence list of block ids. Decode reads `block_table[seq][pos/256]`; the compressors' partial groups
and the index keys become per-sequence as well.

### 3. Scheduler

Continuous batching between the engine and the request queue:

* a step works on a set of sequences: new ones read a prefill chunk (`--prefill-chunk`, e.g. 2048 tokens),
  running ones decode one token;
* a per-step token budget keeps the prefill chunk from starving decode;
* admission order: shortest waiting prompt first (the server already implements this for its FIFO,
  `STRATA_ADMIT_SHORT_FIRST`);
* preemption: when KV is short, park the longest-idle sequence (it already has save/restore code).

### 4. Kernels

This is the hard part, in order of value:

1. **varlen attention** — `ds4_attn_split.cuh` / `ds4_kernels.cu` currently assume one sequence's positions.
   A batched variant takes `seq_ids`, per-seq positions and block tables, and runs all queries of the step
   in one launch.
2. **batched MoE** — routing already produces `(row, expert)` pairs for one sequence. For a batch they are
   concatenated across rows, then a grouped GEMM runs each expert on its rows (the same structure the
   prefill path uses for many tokens, so the pieces exist: `ds4_tq.cu`, `ds4_iq.cu`).
3. **sampling** — one sampler state per sequence (temperature, top-p, seed), streamed per sequence.

### 5. Protocol and server

`GEN` grows an optional sequence id (`GEN <seq> <max_new> …`) or a batch command (`BATCH <n>` then n
requests); `T`/`DONE`/`ERR` carry the id. The server keeps N in-flight requests, writes each request's
lines to the engine as it arrives, reads interleaved token lines, and streams each response to its own
HTTP client. Cancellation (`STOP <seq>`) frees the sequence's blocks.

## Milestones

| # | what | result |
|---|---|---|
| M0 | this document | scope agreed |
| M1 | `SeqState` + scheduler loop, one kernel call per sequence per step | plumbing proven; **no speedup yet** (same total work) |
| M2 | varlen attention | decode overlap of 2-3 sequences |
| M3 | batched MoE | the 2-3x aggregate, first useful version |
| M4 | server streaming + cancellation | usable over HTTP |
| M5 | tuning (DSpark per sequence or off in batch, chunk sizes) | numbers |

M1 is a week of careful refactoring, M2-M3 are CUDA kernel work (4-8 weeks for one experienced engineer),
M4 a week. The risk is concentrated in M2/M3: positions, block tables and expert gather are easy to get
*almost* right, and a wrong row routing shows as a subtly wrong answer, not a crash. The existing parity
tests (`ds4_tq_test`, `serve/test_dsv41.py`) are the model for the new ones.

## Memory

Per-sequence KV at 8K tokens is small (fp4, MLA compressed); the 1M-context configuration reported
`kv_resident: 1048576` with ~2.5 GB for the single sequence. Three sequences at 8K each add well under
1 GB, so VRAM is not the limit for 2-3 users — the limit is the kernels.

## What helps today, without waiting

* the server's short-first admission (`STRATA_ADMIT_SHORT_FIRST=2048`): a quick chat no longer waits
  behind a queued 40K-token agent prompt;
* stable system prompts (cache hits turn into instant prefill);
* `--experts 5` when throughput matters more than the last bit of quality (+5-10 %).

## Open questions

1. Should the first version batch only **decode** steps (prefill stays one sequence at a time)? That is
   much smaller and captures most of the win for chat.
2. DSpark drafting with a batch: one drafter per sequence, or drafting off while the batch is wider than 1?
3. `--parallel N` in the server currently warns and is ignored; the scheduler in M4 should own that flag.

## Path B: burst interleaving with the existing conversation slots (no kernels)

`Model::slot_save()` / `slot_load()` already serialize the whole conversation: its tokens, every layer's
128-token window ring, the compressors' partial groups, the compressed KV and the index keys. Two requests can
therefore take turns on the same model, each with its state parked in host memory - the same mechanism the
server's conversation slots use, but kept in RAM instead of on disk.

Shape: each request runs a burst of K decoded tokens, then the engine saves its state, loads the next
request's, and continues. Every user sees progress every `seqlens` bursts instead of waiting for the others to
finish. It does not raise throughput: the same tokens are decoded by the same kernels, plus the switch cost.

Cost per switch (measured components):

* the state size grows with the context: tokens (4 B/token), the window rings (fixed per layer) and the
  compressed KV plus index keys (proportional to the number of tokens). At 8K tokens this is tens of MB; at 1M
  it is not switchable per burst.
* `slot_save` copies from the devices to host (`cudaMemcpyDeviceToHost` per layer and per source), and
  `slot_load` copies back - roughly `2 x state bytes` per switch over PCIe.
* with a 64-token burst and a 30 MB state, the copy is ~6 ms against ~0.9 s of decoding - about 0.7 %.

So burst interleaving is viable for chat-sized contexts (say up to 32K) and 2-3 users, and it is what the
proposal in this document lists as the low-cost compromise: better perceived latency, unchanged aggregate
throughput. It is not a substitute for M2/M3.

Implementation sketch (not started): the serve loop keeps a `BurstQueue` of admitted requests; after K tokens it
calls `slot_save` into a `std::vector<uint8_t>` keyed by request id, pops the next request, `slot_load`s it, and
continues streaming `T` lines for that request. The scheduler and the KV block tables in this branch are not
needed for it; they belong to the real batching path.

## B2 integration points (server side, no engine change)

`serve/server.py` already loops over generation passes in `Service.run()` (`while True:` near line 5251): each pass
calls `self.engine.generate(prompt, max_new - n, sampling, cancel, embeddings=emb)` (line ~5262) and the loop
continues with a new `prompt` for reasoning recovery and tool calls.  A chunked pass needs three small things
there, behind an env flag (`STRATA_BURST`, 0 = off, default):

1. cap the pass: `per_pass = min(max_new - n, burst)` and pass that to `generate()`;
2. when a pass ends with the model still going (no stop token, no recovery, no tool call), continue the outer
   loop with `prompt = ids + raw_ids` - the continuation the engine reuses (it skips the shared prefix);
3. between passes the FIFO is released, so another request takes the engine: that is the interleaving.

Accounting that must not move: `n` (tokens in this reply), `raw_ids` (every id the model wrote), `segments`
(one entry per pass, used by the DONE statistics) and `first_engine_token_s`.  A chunk boundary is invisible to
the client: the same `T` lines stream out, only later.

Harness: `python -m serve.server --engine mock --port 8099` runs the whole server with a scripted engine and no
GPU, so two concurrent clients can prove the interleaving without touching the production model.

## M4 protocol: what the engine speaks for batch_slots > 0

The server's batched path (`Engine.generate_batched`, serve/server.py:1772) turns on when the engine's INFO block
reports `batch_slots`.  The lines, as the server actually reads them:

* `BGEN <slot> <max_new><keys> <ids>` - one request admitted into a slot (`BGENI` the same, with a DSI1 pictures
  file).  The engine reads the prompt and samples its first token exactly as a `GEN 1` / `GENI 1` would: the usual
  `PP ...`, `T <id>` and `DONE ...` on the control lines;
* then `BADM <slot> <1|0>`: 1 = the slot decodes the rest, 0 = that one token was the whole answer (EOS, max_new 1, a
  cancel);
* the slot's own tokens are `BT <slot> <id>` and it ends with `BDONE <slot> <produced> <finish> <ms>`; `<produced>`
  counts the admission's token too.  Both go to a per-slot queue (`serve/server.py:_pump`), not the control lines;
* `BSTOP <slot>` ends that slot (`BDONE ... cancel`); `BYIELD <slot>` (a much shorter prompt waits for the control
  lines) ends a prompt read between two 1024-token slices - `YIELDED <slot> <tokens read>`, then DONE - and the same
  prompt's next admission reads on from there; `STOP` still stops a solo request and `QUIT` ends the engine.

A slot is a conversation the engine holds between calls (its tokens, KV, samplers).  Advertising `batch_slots N`
**before** the engine speaks BGEN would break every request: the server would send BGEN and the engine would answer
`ERR unsupported command`.  `dsv41 --batch N` says `INFO batch_slots=N` and speaks all of the above; without the
flag nothing changes - the one-at-a-time engine the server ran before.

**The slot keeps its conversation (built).**  A slot that finishes an answer sets its state aside in its own bytes (up to
64k tokens); a later admission whose prompt shares at least 16 tokens with it loads it back before the prefix logic, so
the read stops at that conversation's checkpoint and goes on.  The match is made over every slot's kept conversation -
a re-rendered turn drops the thinking, so the kept state can be longer than the new prompt.

**N slots without kernels (built).**  The model still holds one conversation, so the serve loop parks a slot that is not
decoding in host memory - `Model::slot_save`, the same bytes the conversation folder keeps - and loads it back for that
slot's next burst (`--slot-burst K`, 16 by default).  The slots take turns round robin and their `BT` lines interleave
in the stream.  Measured on the box (2 slots, burst 8, three clients asking 120/60/40 tokens): the two slots' answer
tokens alternate (8 handovers over 47 answer tokens, against 2 with one slot), the reference prompt admitted through
BGEN behind them returns the byte-identical production answer, and a client that leaves mid-stream is stopped with
`BSTOP` without disturbing the next request.  The wall time does not drop (4.13 s with one slot, 4.23 s with two): the
same tokens go through the same kernels - this is the latency half, not the throughput half.

A read gives way to a shorter request (#656): with `--batch N` a prompt is fed in 1024-token slices, and a `BYIELD`
between two of them ends the read - the part read stays in the model, the server sends the same prompt again, and its
read goes on from where it stopped.  Measured: a 4,031-token prompt with a short request 1.2 s behind it - the short
one's first token at 0.77 s instead of 5.47 s, and the engine's own line "the prompt read gave way after 1024 of 4031
tokens".

Between two slices the read shares the engine with the slots that decode: its state is set aside (`Model::slot_save`),
the active slots get bursts for about half the slice's time, and the state is loaded back.  With that the short
request's whole answer took 2.59 s (against 5.70 s with neither mechanism and 5.98 s with the yield alone) - it
finishes while the 4k-token prompt is still being read.

**What a turn costs (M5, measured).**  `STRATA_DS4_SLOT_TRACE=1` prints, per burst, the tokens, the burst's ms and the
switch's ms (the park and the load of the state), and per read slice its size and cost.  With three clients on two slots
and short conversations, a switch costs **~4.4-4.5 ms**; a burst of 8 tokens (~120 ms of decode) therefore pays ~3.7 %,
the default 16 ~1.9 %, and 32 ~0.9 %.  The wall time is flat across those (4.1-5.1 s run to run - three short clients
are noisy), while the interleaving drops with the burst (8 handovers at burst 8, 2 at burst 32): exactly the trade the
design predicts, with no throughput in it.  The switch's cost follows the state's size, so it grows with the context;
that, not the kernel time, is what caps the parked-state approach at chat-sized conversations.

**Where the switch's time goes (measured, `STRATA_DS4_PARK_DEVICE=1`).**  `slot_save/slot_load` copy the state to the host
and back (~4.5 ms for a short conversation's few hundred tokens); `Model::park_save/park_load` keep the same segments -
every window ring, the compressors' partial groups and each source's compressed KV and index keys, per device and per
tensor-parallel member - in per-device arenas, copied device to device, with the tokens and the checkpoints beside them.
On the box (2 slots, burst 8) a switch then takes **2.62 ms** instead of 4.5, with the probe and the three-client check
still green.  What is left is the stream sync and the checkpoints' host copies; a move-based park would halve that.  The
host path stays the default (nothing changes without the env).

**DSpark in a batch (measured).**  The design's open question was whether the drafter should stay on when several sequences
share the model ("likely off").  `STRATA_DS4_BATCH_NOSPEC=1` turns it off for batch slots only, and the answer is the
opposite: three clients on two slots took 4.27 s and 4.39 s with the drafter, 5.09 s and 5.31 s without it (same
interleaving, 8 handovers) - so it stays on at this size.  Worth measuring again with more slots or longer answers.

Known limits of this stage: a read whose state is over 32k tokens is not interleaved (the save/load per slice would
cost more than it gives), a request with pictures runs to its end without being parked (its state cannot be set
aside), and each parked slot holds its conversation in host memory, so the cost per switch grows with the context
(tens of MB at 8K tokens).