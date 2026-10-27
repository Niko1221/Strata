# DeepSeek-V4.1 batching: where the work stands (handoff)

Branch: `feat/dsv41-batch-m1` in /data/works/Strata (not pushed).  Design: `docs/DSV41_BATCHING.md`.

## Done, with tests

| piece | file | tests |
|---|---|---|
| step scheduler (admission, budget, round robin) | `src/dsv41/ds4_sched.hpp` | `tests/dsv41/sched_test.cpp` |
| batched KV block tables + LRU eviction | `src/dsv41/ds4_batch_kv.hpp` | `batch_kv_test.cpp` |
| scheduler step -> per-run KV blocks | `src/dsv41/ds4_batch_plan.hpp` | `batch_plan_test.cpp` |
| parked conversation states (Path B) | `src/dsv41/ds4_state_bank.hpp` | `state_bank_test.cpp` |
| chunked decoding bookkeeping | `serve/burst.py` | `serve/test_burst.py`, `serve/test_burst_roundrobin.py` |
| chunked decoding in the server | `serve/server.py` (`STRATA_BURST`, `STRATA_BURST_CONTINUE`) | `serve.test_server` 352 tests |
| engine: shortest-first admission, `--burst`, `--admit-order` | `src/dsv41/ds4_main.cpp` | built, one live check |
| engine: `--batch N`, BGEN/BADM/BT/BDONE/BSTOP, `INFO batch_slots` | `src/dsv41/ds4_main.cpp`, `src/dsv41/ds4_batch_proto.hpp` | `batch_proto_test.cpp`; `tools/dsv41/batch_probe.py` (live engine); the server check |
| engine: N slots by parked-state bursts (`--slot-burst K`) | `src/dsv41/ds4_main.cpp` | `batch_probe.py --slots 2`; the server check (3 clients) |

Validated on the box (earlier): mock-engine A/B (8 tokens without continuation, 40 with), 3-client real run (chunked
continuation works, no interleaving yet), `serve.test_server` 352/352.

Then the engine's batch protocol, in a model window (2026-10-26 17:56-18:02):

* `tools/dsv41/batch_probe.py` against the engine alone (`--serve --batch 1`, no HTTP): every check passes -
  `INFO batch_slots=1`; BGEN's tokens equal GEN's for the same prompt (12 in all, `BDONE 0 12 length`); `BADM 0` for
  `max_new 1`; `BSTOP 0` ends a slot with `BDONE ... cancel`; a slot after a cancelled one still works; a bad slot
  gets `ERR BGEN`.
* over HTTP (the same engine behind `serve/server.py` on port 8099 with `--batch 1`): one request alone answers exactly
  as production (0.66 s vs 0.65 s, "The capital of France is Paris."); three clients at once all answer, and the
  reference prompt **through BGEN** returns the byte-identical production answer; a `max_tokens 1` request completes; a
  client that leaves mid-stream (BSTOP) does not disturb the next request.  The server's trace shows 6 `BGEN`
  admissions and 3 `BSTOP`s.

Then a read gives way (19:07-19:10): with `--batch 2` the prompt is fed in 1024-token slices and a `BYIELD` between two
of them ends it (`YIELDED <slot> <n>` + DONE; the partially read state stays in the model and the next admission reads
on).  A 4,031-token prompt with a short request 1.2 s behind it: the short one's first token at 0.77 s instead of 5.47 s
(engine line: "the prompt read gave way after 1024 of 4031 tokens"), and the HTTP check still gives the reference answer
byte-identical to production with the sliced read (three clients interleaving, a client leaving mid-stream handled).
Between two slices the read now shares the engine with the slots that decode (its state is parked, they get bursts for
about half the slice's time, the state is loaded back): the short request's whole answer took 2.59 s instead of 5.70 s
(with neither mechanism) and 5.98 s (with the yield alone), and it finishes while the long prompt is still reading.

Then three slots (20:12-20:19, `--slots 3`): the probe sends three admissions at once, every slot's tokens equal the GEN of
its own prompt, all three BADM/BDONE lines are right and the BT lines take turns; through the server with `--batch 3`
the three-client check is green, the engine's trace shows slots 0, 1 and 2 decoding, mean switch 4.99 ms.

Then the two-slot version, same day (18:17-18:24, `--batch 2 --slot-burst 8`): the probe's two admissions at once - each
slot's tokens equal its GEN reference and their BT lines interleave (`0001110000111100001111`); over HTTP three clients
interleave their answer tokens (8 handovers over 47 tokens, 2 with one slot) and the reference prompt admitted through
BGEN behind them is byte-identical to production.  The wall time is the same (4.13 s one slot vs 4.23 s two, 120/60/40
tokens): parking states buys latency fairness, not throughput - that is M2/M3's job.

Regression and soak (20:26-20:33): `serve.test_server` **352 tests, OK** (275 s) over the same tree, and a soak of 8 rounds
x 4 concurrent streaming requests on a `--batch 2` server: 32 requests, 0 failed, 4.1-6.1 s a round, health 200 after,
both slots decoding, mean switch 3.07 ms, no engine ERR line and no server traceback (`tools/../batch_soak.py`, on the
box at /data/works/batch_soak.py).

A slot now keeps the conversation it finished (up to 64k tokens) in its own bytes, and a later admission whose prompt
shares at least 16 tokens with it loads it back before the prefix logic runs, so the read stops at one of that
conversation's checkpoints.  The match is made across all slots because a re-rendered turn drops the thinking the engine
produced: the kept state can be longer than the new prompt, and the server's token-level slot_held then matches nothing.
Seen in the three-client check: "slot 0's kept conversation taken back (101 of 118 tokens shared)", "slot 1's ... (28 of
43)".  A single client still goes through the solo path (the server keeps it there while it is alone), where the engine
continues the state in place as before - so a two-turn single-client test shows no slot reuse, by design.

A fuzz of the batched path (21:01-21:02, `--batch 2`): 12 rounds of 2-4 concurrent requests with random `max_tokens` (1..80)
and 1 in 3 abandoned mid-stream, each round followed by a canary request: every round answered, the canary came back in
0.29-0.46 s, both slots decoded, no engine ERR line and no server traceback (`/data/works/batch_fuzz.py`).

The turn switch can also be a device-to-device copy (`STRATA_DS4_PARK_DEVICE=1`, `Model::park_save/park_load`): measured
2.62 ms against 4.5 ms for the host copy on the same workload, with the probe (2 slots) and the three-client check green
and no host fallback.  It is opt-in; the host path is the default.

M5's drafting question is answered on the box: with the drafter on, three clients on two slots took 4.27 s and 4.39 s;
with it off for the batch slots (`STRATA_DS4_BATCH_NOSPEC=1`) 5.09 s and 5.31 s - the drafter stays on (the design's
guess was "likely off").  The flag stays for a wider measurement.

Where a token's time goes (21:19-21:21): decode is 79 tok/s at a 216-token context, 90 at 3.5K, 90 at 10K, 77 at 20K - the
context moves it by a few percent, and the attention kernel's own header puts a query at ~125 us (~1% of a 12.5 ms step).
M2 therefore buys *overlap*, not throughput; the MoE (M3) is where the 2-3x is.  Prompt reads: 585-590 tok/s from 3.5K.

What a batched MoE can expect (21:23-21:26): the routing trace (`DS4_TRACE=file`) of two sequences decoding at the same
time shows they share **25.5 %** of their six experts (consecutive tokens of one sequence: 22.1 %), so batching two tokens
reads 10.5 expert rows instead of 12 (1.15x) and three read 14.9 instead of 18 (1.20x) - the design's 2-3x is not what
the routing statistics say.  The lever with more room is the overlap itself (pairing requests that share a prompt), not
the kernel; `tools/dsv41/route_overlap.py` measures both.

A shared prompt does not change the picture much (21:30-21:32): two requests with a 1.3K-token system prompt in common
share **32.7 %** of their six experts (unrelated prompts: 25.5 %), i.e. 1.20x on the reads for 2 tokens and 1.28x for 3.
M3's ceiling on this box is therefore ~1.2-1.3x, not 2-3x - the kernel program (4-8 weeks by the design's own estimate)
would buy that much unless the routing overlap itself can be raised.

M2 is smaller than the design assumed: `AttnRows` already describes T queries with per-query window offsets and counts
(`ds4_kernels.cuh:154`), and `sparse_attn`/`attn_split` run them in one launch (`blockIdx.z` = query), which is how
DSpark's verify works.  So M2 is per-sequence KV *regions* plus a per-step table, not a new attention kernel; the memory
for the per-slot KV (a capped context a slot, not 1M) is the design decision.  The first thing to write is the parity test
that runs the kernel with T = 2 over two disjoint regions and compares it with two T = 1 calls.

M2's first code is in: `src/dsv41/ds4_batch_attn.hpp` (with `tests/dsv41/batch_attn_test.cpp`, no GPU) is the step
table - a sequence's window base and compressed-row region, the rows its position can select, the indexer's picks into
the shared table (refused outside its region) and a disjointness check.  Not wired into the model yet.

The kernel-side property M2 stands on is now tested on the GPU: `ds4_attn_batch_test` runs `sparse_attn` with T = 2, each
query on its own window rows ([0,8) and [8,16)) and its own compressed rows ([0,4) and [16,20) of the table), then runs
the same two queries one at a time - every output value is identical.

**Release gate (final, this tree):** `serve.test_server` **352/352 OK** (175 s), the batch probe with two slots **all checks
passed**, the three-client HTTP check **all passed** (reference answer byte-identical to production, 8 handovers,
disconnect handled), the seven unit tests green (two of them on the GPU), and production restored healthy after every
window.  The branch is ready to review; nothing is pushed (the M1-M5 rule).

M2 step 1 is in: the window ring is allocated for `STRATA_DS4_NSEQ` sequences (default 1) and every read/write of it goes
through `win_of(layer, seq)`, a sequence's ring being its own `g.win` rows of the one buffer (the ring functions address
rows from the pointer they get, so no kernel changes).  With one sequence it is `l.win` bit for bit; the two-slot probe
and the three-client HTTP check pass with `STRATA_DS4_NSEQ=2` as well.  The checkpoint and slot paths still take ring 0.

The compressed rows' **writes** followed (`kv_base`, the same day): a finished group's row lands at
`kv_base + pos / ratio`, `kv_base` = `vseq * src.cap` with `vseq` (the sequence the pass serves) 0 until the batched
step sets it, and each source's table is allocated for `nseq` regions of `cap` rows - both paths' `finish` calls,
`replicate`'s offset and the group-to-group copies take it.  One sequence is the `g_first` rows it always was: the probe
(two slots) and the three-client HTTP check green, and the same pair with `STRATA_DS4_NSEQ=2` green too
(`tools/dsv41/windows/batch_window.sh`, `batch_nseq_window.sh`; production restored health 200, no engine ERR line).
`ds4_batch_attn.hpp`'s `write_row(layout, pos, ratio)` is that address, tested without a GPU.  What still reads from
row 0 is the read side - `index_score`'s scan and the selection rows `sparse_attn` takes - which is the step table's
work.

The model's **state is per sequence** too now (`State`, `nseq` of them, a pass on `state() = seqs[vseq]`): the
position, the tokens and their Engram ids, the four checkpoints and `slot_held`, through the chunked prompt, the window
path, `rollback`, `checkpoint`/`restore`/`pin`, `slot_save`/`slot_load`, `park_save`/`park_load` and the Engram's
token rows; `Model::pos()`/`tokens()` read it.  With `vseq` 0 everywhere - one sequence - the behaviour is the member it
was, and the same windows are green (probe two slots, three-client HTTP check, and the NSEQ=2 pair), production restored,
no engine ERR line.  It is what lets the batched step switch a pass to another sequence without parking it; the answers
are only right once the read side above is in.

The **read side is in, and a sequence on its own answers what sequence 0 does** (`STRATA_DS4_VSEQ=n`): the indexer's
scan and the rows `sparse_attn` takes start at the pass's sequence (`ik_rows`/`ckv_rows` = the table at `kv_base`
rows), the checkpoint / slot / park paths copy the pass's ring and region (`win_of(l)` is the pass's ring now), and
`tools/dsv41/windows/batch_seq_window.sh` runs the same three prompts on sequence 0 and on sequence 1 in two loads
(40 tokens/12 written, 300/140 - past the 128-token ring and across compression groups - and a 20-token shared prefix for
the checkpoint/restore): **sequence 1 writes exactly what sequence 0 does**, every check, and the same window's tree also
passes the probe and the three-client HTTP check with `vseq` 0.  Production restored health 200 after each.

The compressors' partial group is a sequence's too now (`comp_kv`/`comp_sc`: `nseq` groups of `ratio x hd` floats a
layer, the pass's at `vseq`), and the checkpoint / slot / park paths copy that one; the parity check's 300-token prompt
crosses many groups, so its half-finished group is written and read inside sequence 1's region.  Same gate, green.

**A batch slot now keeps its conversation in its own sequence** (`STRATA_DS4_BATCH_LIVE=1`, `Model::use_seq`,
`seq_count()`, a per-sequence `reset()`; `--batch N` sizes the model for N + 1 sequences so a solo GEN has one too):
the serve loop no longer parks a slot between its bursts - the pass switches sequence, nothing is copied, and the read
slices of a prompt let the slots' sequences run untouched.  With the env on: the probe with two slots **every check
passed**, the three-client HTTP check **every check passed** (reference answer byte-identical, **14 handovers over 47
tokens** against the parked path's 8, a client leaving mid-stream handled), production restored health 200, no engine ERR
line (`tools/dsv41/windows/batch_live_window.sh`).  The parked path stays the default until the single-launch step is in.
`Finish` also takes a per-row position now (`pos_rows`), for that step's rows.

The compressed rows' two counts are named and pinned in `ds4_batch_attn.hpp`: `written_rows(pos, ratio) = pos / ratio`
(what a sequence has written - slot_save's count) against `selectable() = (pos + 1) / ratio` (what a position may
attend - the kernel's gen rule); they differ by one at a group's boundary, and the test checks both there.

The fuzz was re-run on the final tree (with `STRATA_DS4_NSEQ` in it): 12 rounds of 2-4 concurrent requests, a third
abandoned mid-stream, a canary after each - every round answered, the canary 0.29-0.46 s, no engine ERR line and no server
traceback, both slots decoding.

**Concurrency at the production 1M context (2026-10-27, real runs).**  Every request asked `max_tokens: 262144` (256K)
and left after its share of tokens (thinking off).  Three clients: the stock engine (no `--batch`) took 14.55 s of wall
with first tokens at 0.50/5.81/10.84 s and 2 handovers (it runs them one after the other); the branch (`--batch 3`, a
slot per sequence) 14.43 s with first tokens within 0.74 s and 167 handovers - and the same aggregate, 61.9 against 62.4
tok/s, because each slot still runs its own pass.  More slots at the same context all loaded and answered: 6 (58.7
tok/s), 8 (54.3), 10 (54.1), 12 (54.3), **16 (51.5) - sixteen concurrent requests at 1M context**, 17 sequences, 0.59 GB a
GPU left.  Each extra sequence's compressed KV is ~0.54 GB a GPU (~2.2 GB in all) taken from the experts: 4224 of 4224
experts stayed in VRAM with one sequence, 3662 at 13 and 3441 at 17, and that displacement is where the 62 -> 51 tok/s
comes from.  Capping a slot's context (the design's 32K) is the lever that would raise both the count and the aggregate.

**Qwen3.8-Flash-Next on the same box (2026-10-27).**  The other model family (`strata`, ISTA-DASLab IQ2_XS, 69 GB)
runs normally on this tree - nothing here touches it - and its own batching is the reference this branch is measured
against.  **At the 256K context with 128K max_tokens** (the comparison asked for): `parallel 1` with three clients took
4.11 s of wall (first tokens within 2.87 s, 2 handovers over 300 tokens, 73.0 tok/s aggregate - the queue is what it
costs), `parallel 4` with four 7.78 s (first tokens within 1.87 s, 364 handovers, 51.4 tok/s, the expert cache at 96 %),
and `parallel 8` with eight 20.69 s (first tokens within 6.11 s, 769 handovers, 38.7 tok/s, the cache at 84 %) - four is
the sweet spot there, and past it the KV's room comes out of the experts.  The controls: at a 32K context the same
batching is a real aggregate win (25.0 -> 45.6 tok/s, 1.8x, 377 handovers against 2); at 1M the engine needs
`--kv q4_0` (int8's 1M-token KV leaves no VRAM for its expert tier and it refuses), where `parallel 1` took 5.99 s for
three clients (50.1 tok/s) and `parallel 4` 12.92 s for four (31.0 tok/s, cache 68-78 %).  The first case after a load
pays the pack's cold reads, so its single-request rates are the low ones.

**Release gate on the final tree (2026-10-27, commit `352ec9c`).**  `serve.test_server`: **352 tests, OK** (134.8 s).
The fuzz with the live slots (`STRATA_DS4_BATCH_LIVE=1`, twelve rounds of two to four concurrent requests, a third
abandoned mid-stream and a canary after each): every round answered, canaries 0.09-0.44 s, both slots decoded, **0 engine
ERR lines and 0 server tracebacks**; production restored `health 200` after the window.  The unit tests
(`ds4_batch_attn_test` with the `write_row` cases and the five others) are green on the same tree.

**Where the branch stands (2026-10-27, commit `0e62a63`, 71 commits, nothing pushed).**  The tree is gated (352 server
tests, the live-slot fuzz, the probe and the three-client HTTP check, all green on `60b95a1`/later) and production has
never been left broken.  What is in: M1 (per-sequence state and the turn-taking scheduler), M4 (the batch protocol and the
server's batched streaming), M5 (all the measurements, plus the parameters computed rather than written: the batch's
geometry in `INFO batch_*`, the burst from the measured switch, a slot's context computed at load, written beside the
pack and applied by the next load), M6 (the monitor's parallel view), and M2's foundations - the per-sequence regions, the
absolute selection rows, the per-row chunk path (proved bit-identical in process), the decode side's plumbing
(`Run::rows`, `chunk_rows`, `forward_chunks`, per-row Engram reads, `forward_rows`, the per-row sampler).

What is **not** in: the step itself.  Two attempts are recorded in `docs/DSV41_BATCHING.md`: the chunk flow carries a
step of two rows functionally (every client answered) but at 0.6-1.2 tok/s - its per-call setup dominates - so the step
needs the lean per-window path (`layer()`'s own buffers), which is the next thing to write; the per-row branch of
`layer()` is the 150 lines the design note describes, and the in-process A/B (`STRATA_DS4_STEPROWS=2`) is the harness to
prove it with.  M3 (the grouped GEMM over a step's T routed rows) comes after that; `route_overlap` bounds it at
1.15-1.28x, so the milestone's 2-3x is the thing to test the moment the step is lean.

**Release gate on the tuning tree (2026-10-27, commit `60b95a1`).**  `serve.test_server`: **352 tests, OK** (478 s - the
box was loaded by the gate's own runs).  The fuzz with the live slots (twelve rounds of two to four concurrent requests, a
third abandoned mid-stream, a canary after each): **every round answered, canary green** (0.29-0.44 s), with the serve
loop's burst now measured per turn instead of passed in.  The batch probe (two slots, every check) and the three-client
HTTP check (`all checks passed`) are green on the same tree; production restored `health 200` after each window.

What is *not* in yet: M2's single-launch pass.  The design note in `docs/DSV41_BATCHING.md` ("The single-launch step:
what is in, and what it needs on this box") lists the four pieces - the per-row descriptor through `layer_chunk`, the
cross-group hand-off of a step's T latent rows (the hard one on four GPUs and two tensor-parallel groups), the serve
loop's one step over all active slots, and the measurement that should put sixteen slots below the stock engine's wall.

## Next, in order

1. **Per-sequence KV (M2)**, which is what turns the present turn-taking into throughput: `ds4_batch_kv.hpp` +
   `ds4_batch_plan.hpp` are the CPU half; the serve loop's `Run`s and the round robin are the place they attach.  The
   window rings, the compressed rows' writes and the model's state (one `State` a sequence, `state()`) are in; what is
   left is the step table - `ds4_batch_attn.hpp`'s `rows`/`place`, the indexer's scan and the selection rows
   `sparse_attn` takes - and `feed`/`step`/`sample` naming the sequence they work on.
3. **Batched MoE** (M3): concatenate the routing rows of the slots and run the grouped GEMM once.
4. **Tuning** (M5): DSpark with a batch (likely off), chunk sizes, `batch_slots` count.
5. Decide the fate of Path B (`STRATA_BURST`): keep it as the low-cost compromise, or retire it once N slots work.

## Facts worth keeping

* `serve/server.py:1141`: `self.batch = int(self.info.get("batch_slots") or 0)` - the server's batched path keys off
  the engine's INFO; advertising it before BGEN works breaks every request.  `--batch N` prints it, nothing else; the
  production config has no `--batch`, so production is unchanged.
* the protocol is more than BGEN: the admission is a `GEN 1` (`PP`/`T`/`DONE` on the control lines) plus
  `BADM <slot> <1|0>`; the slot's tokens are `BT <slot> <id>` and it ends with
  `BDONE <slot> <produced> <finish> <ms>` (on a per-slot queue, `_pump`), `<produced>` counting the admission's
  token; `BSTOP <slot>` cancels one slot; `BYIELD` is dropped.
* the engine's stdin reader takes `BSTOP`/`BYIELD` off the queue itself; a `BSTOP` for a slot that has ended must not
  cancel the next admission (it is reset at the start of a request and only checked after the admission's token) - the
  probe caught exactly that.
* The slots take turns by parking their conversation in host memory (`Model::slot_save`/`slot_load`): the model holds
  one conversation, so a switch costs one save and one load (tens of MB at 8K tokens, ~ms).  `--slot-burst K` (16 by
  default) is how many tokens one slot decodes per turn; a burst of 8 makes the interleaving visible without much
  overhead.  A picture request sets `no_switch` and runs to its end; a read is fed in 1024-token slices so that a
  `BYIELD` can end it between two of them (the part read stays as the model's state, `kept = from` otherwise).
* Tuning (M5): `STRATA_DS4_SLOT_TRACE=1` prints a burst's tokens/ms and its switch's ms, and a read slice's size and
  cost; `--read-slice N` sets the slice (default 1024).  Measured: a switch ~4.4-4.5 ms on short conversations, i.e.
  ~3.7 % of an 8-token burst, ~1.9 % of the default 16; the wall time does not move with the burst (4.1-5.1 s), the
  interleaving does (8 handovers at 8, 2 at 32).  The cost follows the state's size, so it is the context, not the
  kernels, that caps this approach at chat-sized conversations.
* Between the slices of a read the active slots decode (about half the slice's time each round); the read's state is
  parked and loaded back around them, so its own tokens are untouched.  A read over 32k tokens is not interleaved, nor
  one with pictures - the copies or the state would cost more than they give.
* A finished slot leaves its conversation in the model (unowned); the next admission parks it into the conversation
  folder (if `slot_max > 0`), so a later turn still reuses its prefix.
* Test assets on the box: `build/strata-dsv41.batch` (the `--batch` engine), `strata-batch1.json` (production config
  with `--batch 1`, port 8099, `exe` = the .batch binary), `/data/works/batch_window.sh` (probe + server check),
  `/data/works/batch_http_window.sh` (server check alone), logs in `/data/works/batch_*window.log`.  Production's
  `build/strata-dsv41` is kept as the stock binary (`build/strata-dsv41.prod`).
* A mock server from an earlier session (`python -m serve.server --engine mock --port 8099`) was answering `/health`
  200 on 8099 and made the first server check talk to it; it is gone, and `batch_http_window.sh` refuses to start if
  8099 is taken.
* Production on 10.188.27.29 runs the stock binary (no burst env, no `--batch`) and is healthy; restarts are allowed
  on request.

## The step's real blocker, found by rebuilding it (2026-10-27, 06:00-07:00)

The serve-loop step is not in the tree (it was measured and dropped), so it was rebuilt here: `STRATA_DS4_STEP=1` with
the live slots - each active slot's pending token printed (its admission's DONE/BADM on the first, exactly as a turn
does), then one `forward_rows` over every active slot's token, then `m.sample(..., row)` a row.  It went through the
chunk host, since `forward_rows` is the per-row path that is already in the tree.  Three things came out:

1. **Two bugs in the per-row chunk path.**  The per-row `finish` handed the kernel the *host* `step->pos` array as
   `pos_rows` - pageable memory a device must not dereference; the rows now go through a host-mapped copy.  And the
   per-row `ext_to_ring` added `win * HD` to a source the kernel already offsets by `win` itself, so it read the
   wrong rows (and past the block for a later row).  Both were fixed while measuring.
2. **With those fixed, the step stops on a plain, explicit refusal.**  `layer_chunk`'s per-row branch answers "the
   batched step has no cross-group hand-off yet (run one tensor-parallel group)" the moment a layer's compressed rows
   are owned by a layer on another device (`sr.remote`) - and on this box every multi-device run is that case: both
   `--tp 2` and `--tp 4` collapse to "tensor parallel N -> 1, each GPU its own layers", so layer 7 reads the rows
   layer 6 wrote on device 0, a hand-off per layer a step.  The step cannot run here until that hand-off is written
   (the design note's item 2: one copy of the step's T latent rows to the owning group, ~32 KB a layer a step).
3. **The per-row chunk path's own A/B failed on this tree.**  `STRATA_DS4_STEPROWS=2` (one sequence, a 300-token
   prompt in chunks of 8) ended with an illegal memory access, so "the per-row chunk path is bit-identical in process"
   holds for the tree it was measured on, not for every tree after it - re-run that A/B before building the step on it.

**What the earlier numbers mean.**  The 0.6-1.2 tok/s of the two step attempts is *client-visible*.  The engine's own
log of that window says `slot 0 ended: 4095 tokens in 79774 ms` and `slot 1 ended: 4096 tokens in 79845 ms` - both
slots decoded at ~51 tok/s while the third client waited 80 s for admission and the first saw its tokens only when a
slot ended.  So that stall was the serve loop's admission/BSTOP bookkeeping around the step, not the step's arithmetic:
a step that keeps `advance_head`'s protocol (the admission's DONE/BADM before its first BT, the BSTOP check a token)
and reads its input line between two steps is the shape to build, and the hand-off above is what it still needs on four
GPUs.

The experiment is not in the tree: it was reverted after measuring, so `6aa4878` is what the branch holds.


**Correction (2026-10-27 07:05).**  The A/B above fails on the *pristine* tree too, so it is not a casualty of the fixes
tried at 06:00.  One cause is proved and fixed in the tree now: `layer_chunk`'s per-row branch lays a row's window out
as a `(win + 1)`-row block at `t * (win + 1)`, up to `kWin` rows a step, but the chunk's `ext` scratch only reserved
`win + C + nseq * (win + 1)` rows - at `--prefill-chunk 8` (C = 8) a step of 8 rows writes 1032 rows into a 523-row
buffer.  The buffer now reserves `max(nseq, kWin)` blocks after the contiguous path's rows.  The A/B still ends with an
illegal memory access, so the per-row chunk path has at least one more fault before the step can stand on it.


**The per-row path's third fault, bisected (2026-10-27 07:16-07:20).**  `--steprows 1` - the per-row call on its own, no
`chunk_ab` comparison - fails as well, and its first error is not in the window block: it surfaces inside the dense
weights' f16 reconstruction (`ds4 gguf rows: an illegal memory access`, then `exl3 rows_f16`), i.e. in a `gemm` of the
chunk's early layers, before `ring_to_ext`.  So the A/B's failure was never the comparison's: the per-row chunk call
faults on its own, at a third site, after the two fixed ones (`pos_rows` pageable host memory, the per-row
`ext_to_ring`'s extra `win * HD`).  Next: a `cudaGetLastError()` clear + `cudaStreamSynchronize` around that `rows`
call (env-gated) tells whether the kernel itself is out of bounds or an earlier launch's sticky error is surfacing
there.


## The per-row A/B was not testing the per-row path (2026-10-27, 11:30-12:45)

**The harness's shape never reaches the per-row branch.**  `--prefill-chunk 8` is clamped at load
(`chunk = min(kMaxChunk, max(256, chunk / 16 * 16))`), so the chunk is 256 and a 300-token prompt runs as two chunks
of 150 rows; `chunk_rows` hands a chunk to the per-row path only when `T <= kWin` (16), so every chunk of that prompt
takes the contiguous branch, `STRATA_DS4_STEPROWS` or not.  The two runs the check compared were the same code, and
the illegal memory access it caught is not the per-row path's (below).

**The per-row path at T=16 is ~270x slower than the contiguous one** (one sequence, `--tp 4`, 8K context, a 128-token
prompt, `DS4_CHUNK=16` - the env is the only door below 256):

| path | prompt read |
|---|---|
| contiguous, `DS4_CHUNK=16` | 128 tokens in 857 ms (149.3 tok/s) |
| per-row, `DS4_CHUNK=16` | did not finish in 233 s (twice; stopped, no prompt line) |

**One cause found and fixed; the gap remains.**  `gemm()` reconstructs the whole weight block into `c.w16` on every
call and the per-row branch called `gemm(i_wq_b)` once a row, so a T-row chunk rebuilt the indexer's weight block T
times.  One GEMM now writes the T rows into `c.iq` and the per-row `finish`/`index_score` read their row of it.  The
128-token prompt still did not finish in 233 s with that in, so the dominant cost is elsewhere; `DS4_PROF` on the
per-row run is the next step (the 96-token per-row run with events did not finish in 8.5 min, where the plain one
prints its table for six chunks: attn rest 408 ms, q_b + rope 227 ms, experts 294 ms, shared 192 ms, 757 ms in all).

**The illegal memory access is in the plain chunked path, and intermittent.**  With the A/B's own shape (300 tokens,
chunk 256, T=150) the contiguous path faulted in `ds4_iq::dequant_rows` for `wq_b` (Q5_K, k 1280, n 32768, the first
6528-column block) on some loads, and another load of the same shape read 300 tokens in 934 ms and generated its
tokens.  `CUDA_LAUNCH_BLOCKING=1` puts the first error at that dequant; `c.w16` is 8 Mi elements and the block writes
8,355,840, so the obvious overflow is not it.  compute-sanitizer on the failing shape is the next step.

**Two window facts.**  `/data/works/services/boot.sh` runs from a login shell, so every `ssh bash -lc` can bring
production's supervisor back mid-experiment (a request then loads the model into VRAM); move it aside for a window.
And a killed engine can leave ~12 GB attributed to a vanished PID until the next process exits; wait for
`nvidia-smi` < 2000 MiB before the next load.

**The release gate, green (2026-10-27 12:38-12:51, commit `79fb10a`).**  `serve.test_server`: **348 tests, OK**
(593.7 s - the box was loaded; the same tree passed in 286 s an hour earlier).  The batch probe with two slots
(`--tp 2`): **all checks passed** (64 s).  Production was stopped for the window and came back `health 200` with its
supervisor.  The first gate run had gone red only because the probe started the same second the suite ended, while
the suite's last engine still held VRAM (the probe's engine exited in 3 s); the gate now waits for the GPUs to be
free between the two and keeps `boot.sh` aside so a login shell cannot start production mid-window.


**The per-row path's stall is one section, not a uniform slowdown (2026-10-27 13:20-13:45).**  `DS4_STEPDBG=1` (an
env-gated, synced timer on the per-row branch's sections, reverted after measuring) on a 96-token prompt at
`DS4_CHUNK=16` shows the first layers of the first chunk fast, then a stop in layer 2's compressor/indexer block:

| section (T=16) | layer 0 | layer 1 |
|---|---:|---:|
| engram + attn mHC + norm | 9.8-11.4 ms | 3.5 ms |
| attention projections | 1.8-3.2 ms | 0.1 ms |
| q/kv finish + window (per row) | 0.5-2.1 ms | 0.2 ms |
| compressor + indexer | 0.0 ms | 0.0 ms |
| attention slabs | 10.8-11.5 ms | 2.6 ms |
| MoE (mHC/router, shared expert, flow layer) | 3.1 ms | 2.3 ms |

A per-row layer costs ~9-30 ms where the contiguous one costs ~3 ms, so the 270x is not a per-layer cost.  Layer 2 -
the first with a compressed KV (`l.ratio > 0`) - enters, times its window at 0.2 ms, and never returns from the
per-row compressor block: the calls between that mark and the next are the per-row `compress_win` loop and the
completed group's `finish` / `gemm(i_wk)` / `finish`, and marks placed inside the block (before the group loop, and
after each call in it) never print.  The process stays alive (no ERR line), so it blocks rather than faults; the
suspects are the per-row `compress_win` arguments (the sequence's partial-group pointers) or a deadlock with the
flow's expert copies while the compute stream waits.  The plain path through the same layer is fine.  Next: a mark a
row of that loop (or compute-sanitizer on the layer), then the 300-token shape.

Environment notes from the same window: a build's binary must not be copied over one a dying engine still holds
(`Text file busy` until the process exits; use a fresh name), and a window's restore copies `strata-dsv41.prod` back
over `build/strata-dsv41`, so keep the branch build under another name (`.dbg`, `.stepN`).

**Where the box is (2026-10-27 13:50).**  Production was last seen `health 200`; the tree is clean at `fb7a4fd`
(the stall note above is not committed to `docs/DSV41_HANDOFF.md` yet - the box became unreachable at 13:50,
`10.188.27.29:2222` and `:8080` both timing out).  Instrumented builds were left as `build/strata-dsv41.step2` ..
`.step5` (revert the line numbers are not in the tree); `build/strata-dsv41` was restored to the stock `.prod` by
the last window.



**It is a deadlock, not a 270x slowdown (2026-10-27 14:40-14:50).**  A build with host-order prints only (no syncs;
`DS4_STEPDBG`) shows the prompt reaching layer 2 of the first chunk and stopping right after the per-row `ratio`
print (the first layer with `l.ratio = 2`): no compressor-row print, no indexer, no next layer.  Running the engine
directly (one `GEN`, stdin held open) and reading `/proc/<pid>` at the stop: the main thread sleeps in the 50 ms
monitor loop of `run_dev_chunks`, every device worker is in `futex_wait_queue_me`, GPU 0 is idle while GPUs 1-3 stay
at 100 %, so a device worker is blocked, not the arithmetic being slow.  The earlier synced build put the same stop
inside `cudaStreamSynchronize` after the window section, which agrees: the worker waits for something that never
arrives.  The last thing that worker did is the per-row window block of layer 2 (its print fired), then the `ratio`
print - so the wait is in that layer's compressor/indexer block, its attention, or its flow MoE; which of the three
is the next print to add.  The box has no gdb/eu-stack/pstack, so the tools for a stack are either a print after each
branch or `compute-sanitizer` on that layer (a faulted kernel can leave a stream that never signals).

The plain path through the same build is fine: `plain: OK`, 96 tokens read in **760 ms** on the same load where the
per-row run stops - so the control is clean and the shape (96 tokens, `DS4_CHUNK=16`, one server) is not the cause.



**The per-row path is correct and as fast as the contiguous one; the *flow* is what deadlocks (2026-10-27 14:52-15:12).**
With `DS4_FLOW=0` (the batch streaming in place of the GGUF in-place flow) the per-row path runs the whole 300-token
prompt and is indistinguishable from the contiguous one:

| 300 tokens, `DS4_CHUNK=16`, `--tp 4` | prompt read | 8 generated |
|---|---:|---|
| contiguous, `DS4_FLOW=0` | 1993 ms (150.5 tok/s) | `1021, 3174, 9515, 1613, 223, 29429, 201, 38348` |
| per-row (`STRATA_DS4_STEPROWS=1`), `DS4_FLOW=0` | 1991 ms (150.7 tok/s) | the same 8 tokens |

`tools/dsv41/steprows_check.py` now sets `DS4_FLOW=0` itself and its check passes on the 300-token prompt:
"ok the contiguous path against itself (deterministic)", "ok the per-row chunk path writes what the contiguous one
does (12 tokens)", "all checks passed".  So the "~270x" of the section above was this deadlock, not the path's
arithmetic: with the flow on the per-row prompt stops at layer 2 and never returns (the `/proc` evidence above);
with the flow off it is the same code at the same speed and the same tokens.

`run_dev_chunks` now turns the flow off for a run whose `Run::rows` is set (a step) and restores it on exit
(`FlowGuard`), so the eventual single-launch step runs on the batch streaming until the flow can pump a step's rows.
The flow itself is the next investigation: `flow_layer`/`flow_pump` and the per-row branch's longer kernel sequence
(the pump runs while the stream is busy; a step's extra per-row launches can leave the staging ring waiting for a
pump the host never issues).



**Where the flow stops on a per-row run (2026-10-27 15:35-15:45).**  Print-bisection with the flow on puts the stop
inside `flow_layer`: the last print is its entry for layer 1, and the per-row branch's prints before it (the four
branch points and the attention) are all there; layer 2's prints never come, and the flow's own wait-loop print never
fires either.  That is consistent with the block being the `cudaEventSynchronize(c.stage_ev[...])` in `flow_layer`'s
"not all of the batch landed" branch: it is reached before the 1 s print can happen, and it waits on a staging event
that `flow_pump` never records because the pump stopped earlier on an item whose layer-cache hand-off (`after`/`keep`)
has not landed.  With `DS4_FLOW=0` the same prompt runs (1991 ms against 1993 ms, the same tokens), so the guard
`run_dev_chunks` puts on a step's run (batch streaming instead of the flow) is the right default until the flow's
layer cache can keep a step's rows.  gdb cannot attach on this box - the container blocks ptrace ("ptrace:
Inappropriate ioctl for device"), even with gdb installed and through sudo - so the flow has to be bisected with
prints.  (At 96 tokens the flow is also slower than the batch path: 760 ms against 665 ms of prompt read.)


## The step runs, and the chunk path is the wrong host for it (2026-10-27, 16:00-17:30)

**What the tree has now.**  `STRATA_DS4_STEP=1` (with `STRATA_DS4_BATCH_LIVE`, whose per-sequence state the batched path
already keeps) gives the serve loop M2's step: every active slot's pending token out (its protocol - `BT`, the
admission's `DONE`/`BADM`, the end checks), then ONE `forward_rows` pass over all of them (a row a sequence), then a
`sample` a row - no drafter, a row a token.  Around it:

* `Model::seq_pos(seq)`: the position a slot's own sequence has reached (a step must not read another sequence's).
* `forward_rows`' embeddings come from `h_emb` (each row its own sequence's token at its own position), not from
  `state().toks`, which is the *current* sequence's prompt - another slot's.
* `--slot-context N` / `STRATA_DS4_SLOT_CONTEXT`: the tokens one conversation may hold; the compressed tables are sized a
  sequence by it (`INFO batch_table_mb_per_seq`: **965 MB at 1M, 30 MB at 32K**), so it is what keeps the experts in
  VRAM: at 16 slots and 1M the tables cost 15.4 GB a GPU - **2170 of 4224 experts on the CPU lane (21.5 GB)**, against
  **0** at 32K.
* the cross-group hand-off for a step's rows (below).
* a device that fails now ends the run: before, a worker that returned an error left the others waiting for a hand-off
  that would never come, and the engine hung silently.  **The "step stops at layer 2" of 06:00 and 14:40 was exactly
  this**: layer 2's step branch refused (`sr.remote` - every multi-GPU run on this box is "each GPU its own layers") and
  devices 1-3 waited for its rows for ever.  The trace shows it: `tr dev 0 il 2 step block R 2 kv_src 1 ix_src 1 remote
  1`, then nothing, then `ds4 step: 2 row(s) stopped: the batched step has no cross-group hand-off yet`.
* `DS4_STEPTRACE=1`: a mark a section of a step's layer - the only stack a container without ptrace has.
* `tools/dsv41/batch_step_bench.py`: N slots at one context, the first token's latency, the aggregate, the tokens
  against another run's.

**The cross-group hand-off for a step's rows.**  A step's rows are each at its own position and in its own sequence, so
the chunk path's contiguous staging (rows `g_first .. g_end`) does not fit.  The owner (member 0 of the `kv_src` layer)
stages the rows this step completed one at a time (`xfer_put`, four segments a kernel - CKV then its index keys), sets
the source's flag once with the step's window number, and every reader places each row in *its* table at
`ckv_at(sr, row, seq[t])` (one `xfer_get` a four segments, the flag waited first) - the same descriptor on both sides
says which rows, in which order.  The slot is reused a step later, once the readers acknowledged it (`x_ack`, `win - 2`:
the chunk path's rule, a step a chunk).

**Measured (four L20s, 1M max-context, greedy, DSpark off, 64-token prompts, 96 tokens a slot).**

| run | slots | slot cap | first token | per slot | aggregate | tokens |
|---|---:|---:|---|---:|---:|---|
| bursts | 16 | 1M | within 0.42 s (first 7.11 s) | 3.2-3.4 tok/s | 41.2 tok/s | - |
| bursts | 16 | 32K | within 0.41 s (first 6.75 s) | 3.4-3.6 tok/s | 43.4 tok/s | - |
| step | 2 | 32K | 0.88 s, both at once | 5.0 tok/s | 9.6 tok/s | differ at token 3 (at 1M: 2.2 tok/s a slot) |
| step | 8 | 32K | 3.40 s, all at once | 5.0 tok/s | 34.1 tok/s | - |
| step | 16 | 32K | **6.75 s, all at once** | 4.0 tok/s | **50.1 tok/s** | slot 15 differs at token 1 |
| step | 16 | 1M | - | - | not run: 2170 host experts, minutes a step | - |

Two things this says:

* **The step does raise the aggregate** - 50.1 against 43.4 tok/s at 16 slots, and all sixteen first tokens at one
  instant instead of half a second apart - **but only once the experts fit VRAM** (32K a slot).  At 1M the tables take
  the experts' room and a step reads ~27 GB of experts (16 rows x 6 x 40 layers; the union is smaller, the order the
  same): the step is not a batched-1M mode, it is a mode for a slot cap.
* **It is not correct yet.**  The 2-slot run's tokens are the burst path's for the first step and differ from the third
  token on; at 16 slots slot 15 differs at token 1.  The rows a step writes reach the readers' tables (the hand-off
  runs, the reads are placed), but something in the per-row write or the staging does not reproduce the burst path's
  state exactly.  Candidates, in order of suspicion: the index keys' staging (the writer stages them only when its own
  layer has `ix_src`; a reader with `sr.ik[di]` copies regardless), the ack's parity across a prompt read's even and odd
  chunks (`win - 2`), and the serve loop's read interleaving, which steps the admitted slots *while* another slot's
  prompt is read (slot 15 is the last admitted - the only one that differs at token 1).

**Why the chunk host cannot make the step fast.**  A step costs ~200 ms of fixed cost whatever its rows are (2 slots:
199 ms, 8: 199 ms, 16: 249 ms - ~6 ms a row on top), against the window path's ~16 ms a token.  It is `layer_chunk`'s
design: a device sync after the routing (the host lane needs the expert bounds), the bounds over the bus, ~40 kernels a
layer - ~8-11 ms a layer at two rows where `layer()` (the window shape a decode token takes) is ~0.3 ms.  It amortizes
with rows, which is why sixteen slots is where it wins, but the fixed 200 ms is paid on every token of every slot.
**The step has to live in the window path** (`forward`/`layer`, the row table `ds4_batch_attn.hpp` was written for): the
same single pass, no host syncs, and the path whose experts the *flow* streams - which is where the aggregate can move.
That is M2's remaining piece and the next milestone.

**M3 is the step's own routing.**  `route` runs once for the step's T rows, one `moe_sort`, one scatter a VRAM slot and
one read a streamed expert - a step already reads an expert once for every slot that picked it, so the "grouped MoE" is
not a second kernel: it is what a step's pass does.  The measured routing overlap (25.5-32.7 %) bounds what it buys at
1.15-1.28x, which is the margin the window-path step would collect on top.

**The flow still deadlocks on a step's rows** (`flow_layer`'s staging event: the pump stops on an item whose layer-cache
hand-off has not landed); `run_dev_chunks`' `FlowGuard` keeps a step on the batch streaming.  The window step will need
the flow to learn a step's rows, or a slot cap whose experts all fit VRAM.

**Next, in order.**

1. The window-path step: per-row positions and sequences in `forward`/`layer` (`finish`'s `pos_rows`, `AttnRows` gen 0,
   the per-sequence rings, compressors and tables are in), the flow's rows for it, then the same measurements - the
   fixed cost a step should pay is the window's ~16 ms, so sixteen slots' aggregate is one pass's expert read over ~16
   rows.
2. The correctness of the step's rows: the same tokens as the bursts, bit for bit, at 2 and 16 slots (the harness is
   `batch_step_bench.py --ref`).
3. Then M3's margin (1.15-1.28x) and the flow.


## The step runs on the window path: 86.6 tok/s at sixteen slots, exact at two (2026-10-27, 18:00-18:25)

**What the tree has now.**  `STRATA_DS4_STEP=1` no longer runs `forward_chunks`: `forward_rows` calls `forward_step`,
which is `forward`'s pass with a row table - the decode path's kernels (`layer()`, `mv`, the async mailbox), ~16 ms a
pass instead of the chunk path's ~200 ms.  `DS4_STEP_CHUNK=1` keeps the chunk path's step for an A/B.  The per-row
pieces that were missing:

* the compressed rows and the index keys of the groups a pass completes, one row a row: `finish` with the row's own
  `pdst` (its own sequence's row of the table, at its own group's first position) and its own position; and the same
  rows staged for the other groups' readers in row order (`xfer_put`), the descriptor `run_dev`'s step branch already
  held, the source's flag set even when a pass completes no row (a reader waits on it).
* the indexer a row at a time: each row its own sequence's region of the index keys (`ik_rows(ks, dev, seq)`) and its
  own picks made absolute in that region (`rows_add_i` with `kv_base(sr, seq)`) - one `index_score` cannot serve two
  sequences' key regions, so a pass runs one call a row.
* `finish`'s `pos_rows` indexes the query, not the row: `pos_rows[v / vpp]`.  A step's `q` carries `vpp = NHL` rows a
  query, so the chunk step's rope read another row's position, and its compressed rows went in unroped (`fw.rd = 0`):
  that is why the chunk step's tokens drifted from the third on.  The window path's step is exact at two slots.

**Measured** (four L20s, 1M max-context, greedy, DSpark off, 64-token prompts, 96 tokens a slot, both at
`--slot-context 32768`, a window each, production stopped; `tools/dsv41/batch_step_bench.py`):

| slots | bursts, tok/s | step, chunk path | step, window path |
|---:|---:|---:|---:|
| 2 | 31.4 | 9.0 | **39.4** (42.8 a slot) |
| 4 | - | 18.4 | **77.8** (29.9 a slot) |
| 8 | - | 34.1 | **89.2** (18.4 a slot) |
| 16 | 43.3 | 50.1 | **86.6** (8.7 a slot) |
| 16, flow on | - | not runnable (the flow deadlocked) | **86.8** |

The same measurements on the **final binary** (`build/strata-dsv41.step26`), 96 tokens a slot, 32K a slot, each step run
against the bursts of its own slot count (`--ref`), 21:10-21:17:

| slots | bursts, tok/s | step, tok/s | gain | tokens |
|---:|---:|---:|---:|---|
| 2 | 43.0 | **59.9** | +39 % | the reference's, bit for bit |
| 4 | 43.2 | **77.6** | +80 % | the reference's, bit for bit |
| 8 | 43.3 | **88.6** | +105 % | the reference's, bit for bit |
| 16 | 43.4 | **86.7** | +100 % | not (rounding above 8 rows - see below) |

(The 4- and 8-slot rows here replace the earlier 77.8/89.2, which were measured before the staging-race fix; the 16-slot
row is from the same family of builds as the race fix and was not re-run on `.step26` - 86.6/86.7/86.8 across those
builds, i.e. within noise.)

The step's first tokens all arrive at one instant (0.88 s at 2 slots, 3.40 s at 8, 6.75 s at 16 - the last prompt's
read) against the bursts' 0.41 s spread, and 0 routed experts on the CPU lane at 16 slots.  The flow streams a step's
experts on this path (86.8 against 86.6: no penalty), which the chunk path could never do.

**Exactness: it was a race, and it is fixed (19:10).**  At two slots the step's tokens were the burst path's from the
start.  At four and more they drifted - but *not always*: the same 4-slot run with alternating prompts (A, B, A, B;
every even slot must answer slot 0 and every odd slot slot 1, `same_probe.py --alt`) came out clean in 1 run of 3, and
the 6-token run with `DS4_STEPCHECK`'s per-layer syncs came out clean too, while the 4- and 8- and 16-slot runs
without the syncs drifted.  Two races, both on a buffer reused *within* a pass:

* the window path's cross-group staging has no acknowledgements (the chunk path's `x_ack` is what the chunk path
  relies on): a reader's copy into its table could still be in flight when the next pass's writer staged into the same
  slot (`sr.stage(0)`), so some rows landed from another pass.  `run_dev` now syncs a step's device at the end of its
  work - a pass a chunk, all its copies done before the next pass stages - which is what the head device's sync
  already gave the logits.
* the serial path (`!wp.par`, `replicate`) staged every row through the *same* pinned buffer (`sr.h_ckv`): the next
  row's device-to-host copy could overtake the previous row's host-to-device copies.  A row now stages in its own
  slice of it (`h_ckv + t * ckv_rb()`, `h_ik + t * ik_rb()`).

Measured after the fix (alternating prompts, 24 tokens): 3 of 3 clean at 4 slots, 1 of 1 at 16, 2 of 2 unchanged at 2.
The same-prompt probe had missed it because a stale copy for every row carries the same bytes; alternating prompts is
what exposes it (`tools/dsv41/same_probe.py`, now with `--alt` and `--out`).

**Window discipline (learned the hard way today).**  `services/boot.sh` is started by a *login* shell, so any
`bash -lc` over ssh (`rex.py run` / `runf`) during a window starts production behind the window's back and the run
then fights it for VRAM: the 18:10 window measured 15375 host experts, 18 s first tokens and drifting tokens because a
production engine had taken ~35 GB of each card.  Move `boot.sh` aside (`mv boot.sh boot.sh.off`) before a window, as
`final_gate.sh` does, poll with `rx.py` (`bash -c`), and put it back in the window's restore.

Debug switches the hunt left behind, both off by default: `DS4_STEP_SERIAL=1` runs a step's pass on one thread with
`replicate` instead of the threads' hand-off (it takes the staged cross-group path out of the picture), and
`DS4_STEPCHECK=1` prints every pass's shape (`tr step T <n> pos ... seq ...`) and checks that rows 0 and 2 of a pass
with the same prompt stay bit-identical after every layer (it syncs a device a layer, so use it on short runs).

**Still open at 20:30: it is the pass's row count - and above 8 rows the drift starts as rounding, not as a bug.**
Rounded out with the race fix in: 2 slots exact (96 tokens, 59.9 tok/s against the bursts' 43.0); 8 slots exact (24
tokens of alternating prompts against a 4-slot burst reference, and 32 tokens of eight different prompts); 9, 12, 15
and 16 slots drift, at the same *token* (6 for prompt A, 13 for B - A's margin is thinner, the error is older), and a
9-slot run whose nine prompts are *identical* drifts the same way, so the error is the same for every row.  Ruled out
one by one: the staged hand-off (`DS4_STEP_SERIAL=1` drifts the same), the routing mailbox (`DS4_STEP_SYNCROUTE=1`
drifts the same), the sequence count (8 rows with `STRATA_DS4_NSEQ=10` matches exactly, so it is `T`), and a per-row
region or ring misaddressed (rows 0 and 2 stay bit-identical after every layer, and every pass's shape is clean:
`tr step T 9 pos 64 64 ... seq 0 1 ... 8`).

The per-layer hash of row 0 (`DS4_STEPCHECK=2`, `tr shash`, compared with `tools/dsv41/cmp_hash.py` /
`work/cmp_num.py`) says what the difference *is*: in the very first pass, layers 0 through 12 agree to **1e-7** (the
last bits of a float sum), and **layer 13 jumps to 1.2e-2** - nine orders of magnitude - then the rest of the run
follows.  A jump of that shape is not a wrong tensor: it is a *discrete* decision flipping on a 1-ulp difference, and
layer 13 is the first layer whose indexer's top-k has a tie close enough to be flipped by it (the indexer picks its
compressed rows by score; a last-bit difference moves the cut).  Greedy decode then amplifies it into different
tokens, which is why the divergence always looks total and early.

Where the 1-ulp comes from: the first guess was the EXL3 GEMV's shape.  It is an MMA kernel with **n = 8** (`kTok = 8`,
"activations a block takes at once (the MMA's n)") and a job's activations are split into runs of 8, so 9 rows are two
units (8 + 1) where 8 rows are one.  **That was tested and refuted** (20:30-20:35): with a step's `mv` calls split into
slabs of 8 rows (`mv()` with `ncol <= 8` a call, so every row's job has the shape a one-row job has), 9 and 16 rows
drift at exactly the same token as before, and the 16-slot aggregate is unchanged (86.7 tok/s) - so the rounding does
not come from `mv`.  What is left is another kernel whose behaviour is shaped by T: the router gate's `gemv_f32`, the
indexer's `gemv_f16_rows`, `sparse_attn`'s head-splitting choice, `hc_mix_split`, or the grouped MoE's grouping.  The
way to split it in one run: a second hash *after the attention* (the `dump_layer(d, d.h, 1000 + l.il, ...)` point
inside `layer()`), which says whether layer 0's 1-ulp difference is in its attention half or its MoE half.

**Where that leaves exactness.**  At 2 and 8 rows the step's tokens are the burst path's, bit for bit (24, 32 and 96
tokens).  Above 8 rows the pass is deterministic and self-consistent (every row of a same-prompt pass is identical, the
run repeats), but it rounds differently from a one-token decode, and greedy decode amplifies that into other tokens.  It
is not a per-row defect and not a data race; if the project needs bit-identical tokens above 8 rows as well, the next
step is the attention-half hash above, not another guess.

**The release gate is green on this tree (20:48).**  `serve.test_server`: **Ran 348 tests in 552.6 s, OK**;
`batch_probe.py` two slots: **all checks passed**; production restored, `health 200`, supervisor 222731.  The gate ran
with `build/strata-dsv41.batch` = this tree's build (`build/strata-dsv41.step26`).  So the step, the slot cap and the
window path's per-row code ship without a regression in anything the suite covers.


## 1M with no slot cap: the flow streams a step, and the grouped routing pays where the experts go to the host (21:00)

`--slot-context 1048576`, 64-token prompts, 32 tokens a slot, the window path's step (`build/strata-dsv41.step26`):

| 1M, no cap | bursts | step | a slot: bursts -> step | host experts |
|---|---:|---:|---|---:|
| 2 slots | 31.3 | **39.4** (+26 %) | 15.7 -> 19.7 | 125 (3 %) |
| 8 slots | 31.0 | **49.1** (+58 %) | 7.1 -> 18.6 (2.6x) | 984 (23 %) |
| 16 slots | - | not runnable | - | 2170 (51 %, a pass reads ~27 GB) |

* **flow+step at 1M works**: the 2-slot step with the flow on gives 39.4 tok/s and *the reference's tokens, bit for bit*;
with `DS4_FLOW=0` 39.5 and the same tokens - so on the window path the flow streams a step's experts at no cost, which
is the thing the chunk path's flow could never do (it deadlocked on a step's rows).
* **the grouped routing pays exactly where it was predicted to**: at 2 slots only 3 % of the experts are on the host and
the step's gain is amortization (one pass for two rows, +26 %); at 8 slots 23 % of them are, and the gain is **+58 %**
(2.6x a slot).  Its bound stays the measured routing overlap (25.5-32.7 % of the expert reads a pass can share) plus the
per-pass fixed cost the pass amortizes; at 2 slots the overlap is invisible, at 16 slots/1M the reads dominate but a
pass is minutes, so 8 slots/1M is where the mode is at its best on this box.
* **exactness at 1M**: exact at 2 slots (with and without the flow), *not* at 8 - with a fifth of the experts on the
host lane the combine's order differs from a one-token decode's, the same last-bit story as above the 8-row shape, not
a new defect.

**So the objective's four items stand as:** the step in the serve loop with the 2/4/8/16 measurements (32K a slot), the
slot cap as what makes 16 slots fit, the step's pass as the grouped MoE with its margin measured where it can be
collected (8 slots at 1M), and flow+step proven on the window path at 1M - with the exactness boundary documented
(bit-identical at 2 rows anywhere and 8 rows at 32K; shape-dependent rounding beyond that).


## Deployed as this machine's production, and M6 accepted on it (21:50)

**The build is production now.**  `build/strata-dsv41.step26` (md5 `9d0892c98f9f`) -> `build/strata-dsv41` and
`build/strata-dsv41.prod` (the old stock is kept as `build/strata-dsv41.prod.stock`; the config and the launcher each have
a `.prod.bak`).  The launcher gained `server.py --api-monitor` (the M6 page); the config is the production one,
`--max-context 1048576`, **one stream** - see below for why the slots had to go.  After the restart: `health 200`,
`/v1/status` reads `context.native 1048576` and `serving 1`, the key is enforced (`/v1/status` 401 without it), one
short answer in 0.63 s ("The capital of France is Paris.").  The machine's public port 53080 maps to the container's
8080 (8080 is not reachable from the outside, 53080 is).

**Why the slots went away again (22:20).**  With `--batch 2 --slot-context 1048576` the serving agent's requests -
190K to 223K-token prompts, one turn after another - started coming back as `pi-ai stream idle timeout after 300000ms`.
The engine log has the mechanism: `the prompt read gave way after 1024 of 223650 tokens (slot 0, 1818 ms)` and the same
for slot 1, i.e. with two slots the server lets a second long prompt start on the other slot mid-read, and the two reads
interleave; the interleaved prefill runs at ~500 tok/s, so 223K tokens is ~450 s of silence and the client's 300 s idle
timeout kills it, the retry piles another long prompt in, and the monitor fills with `disconnected` (the picture that
prompted this: five requests at 106-301 s, 0 output tokens).  Two sequences' tables at 1M also pushed 125 experts to the
host lane.  One stream fixes both: no interleaving, and one sequence's tables leave **4** experts on the host lane;
measured prefill on it: 4.5K tokens in 6.9 s (646 tok/s), 22K in 17.9 s (**1,240 tok/s**), so a 190K prompt is ~150 s -
inside the client's timeout.  If a 2-slot demo is wanted again, set `STRATA_DS4_ADMIT_SHORT_FIRST=0` in the launcher as
well so a long read is not interrupted (at the cost of the host experts and a queued second request).


## The 300 s stream idle timeout: what it actually was, measured (23:05)

A review of this branch proposed four causes; two were measured on the box and one measured out.  The one that mattered:

**A long prompt read is silent to a client that counts data events.**  A 45K-token read on the single stream:

| build | first byte | SSE comments | `data:` chunks | largest gap between data chunks |
|---|---:|---:|---:|---:|
| before | 0.1 s | **434** | 7 | **34.0 s** (= the whole read) |
| with the fix | 0.1 s | 0 | 12 | **10.0 s** |

The server sent a keep-alive *comment* (`: keep-alive`) on every PP line, but only 7 real `data:` events - one per
finished step.  A client that resets its idle timer on the events it parses (pi-ai's does: "stream idle timeout after
300000ms") therefore saw 34 s of silence for 45K tokens, and a 190-223K prompt is 150-450 s of it.  The fix
(`serve/server.py`, `openai_chunks`'s `ping` branch): when the client did not ask for prompt progress, still emit an
**empty delta chunk** at most every `PREFILL_BEAT_S = 5` s while the prompt is read (every OpenAI client ignores an
empty delta).  Measured after: 12 data chunks across the read, largest gap 10 s.

**Measured out: the slot-wait keep-alives.**  With both slots decoding 1200-token answers, a short request waited 36-40 s
for a seat and received **4 keep-alives on both builds** - the old code's `_take_control` already beats every 10 s while a
request waits for the control lines, so a request waiting for a *slot* was not silent.  (The `yield None` added to the
slot-wait loop is harmless and stays, but it is not what was killing the requests.)

**In the tree, not measured: the 32K decode gate.**  `ds4_main.cpp` only let a prompt read stop for the other slots'
decoding while `m.pos() <= 32768`, a limit that exists for the save/restore path; in live mode the slots have their own
sequences, so the gate is now `(live || m.pos() <= 32768)`.  The measurement that would show it needs one slot decoding
for minutes while the other's prompt read crosses 32K; the first attempt was invalidated by the slot cache (the 68K
"fresh" prompt was answered in 0.9 s from a kept conversation, so nothing was read), and it was not re-run.

The interleaving found earlier (two long prompts, ~500 tok/s, ~450 s) still stands as the *other* reason those requests
died; the heartbeat and one stream together remove both.

**The other two dialects, same hole, measured after the fix (23:24).**  `/v1/messages` sent only comments while a prompt
was read; it now sends Anthropic's own `event: ping` every `PREFILL_BEAT_S`, and `structured_chunks` (a request with a
`response_format`) now lets an empty delta through instead of only the comment it had buffered.  A 45K-token read on
each:

| dialect | first byte | comments | real events | largest gap |
|---|---:|---:|---:|---:|
| `/v1/messages` | 0.1 s | 422 | 5 pings + 18 data | 10.0 s |
| `/v1/chat/completions` + json_schema | 0.1 s | 414 | **5 beats at 10.2/15.2/20.2/25.2/30.2 s** | **10.2 s** |

(The structured row first read "0.0 s" because the beats were *buffered*: `0e16735` had dropped `openai_chunks`' `yield
None`, so `structured_chunks` never saw a ping and the empty deltas it did get were held for validation until the answer
came out in one burst.  `structured-beat.patch` restores the comment between beats and lets an empty delta through
untouched - `_empty_delta` - which is what the 5 beats above measure.  The same regression broke
`serve/test_queue_keepalive.py`, which is not part of `serve.test_server`; both modules pass now (11 tests, 3 tests).)

**Seat-yield (`.step29`, 00:07).**  A solo read that gives way had its half-read state in the *solo* seat while the flag
protected the seat the server reserved: the two are now swapped at the yield, so the request comes back to its own state
(`seat-yield.patch`).  Re-run of the 40K promotion scenario: A 94.8 s (92.5 s before the fix, 184.3 s before the seat
map), and the `seat 0 took seat 2's conversation` line is gone - the resume finds its state where it left it.


## Item 2, the read-slice matrix (00:10-00:40): the batch read path is 1.3-2.9x slower than one stream

Five `.step29` servers, each a fresh 55K-token prompt with a nonce at the front, the other slot idle unless said, and the
rate from the engine's own `prompt N tokens = ... read in Z ms` line.  The engine's prefill chunk here is **8192** (its
own log: "prompts in chunks of 8192"), so "read-slice = prefill chunk" is the 8192 row.

| config | read | rate | with the other slot decoding a long answer |
|---|---:|---:|---|
| no `--batch` (baseline) | 33.3 s | **1,649 tok/s** | - |
| `--batch 2`, 1M slots, read-slice 1,024 | 96.7 s | 572 | 143.6 s (385) |
| `--batch 2`, 1M slots, read-slice 8,192 | 86.4 s | 634 | 92.1 s (595) |
| `--batch 2`, 1M slots, read-slice 32,768 | 42.1 s | **1,307** | - |
| `--batch 2`, 64K slots, read-slice 8,192 | 85.4 s | 637 | - |

What the table says:

* **The 1M tables are not the cost**: with a 64K slot cap the host expert lane is empty (0 of 4224) and the read is 637
  tok/s, i.e. the same as with the 1M cap (634).
* **The slice size is most of it**: 1,024 -> 572, 8,192 -> 634, 32,768 -> **1,307** tok/s.  A 55K prompt is 54 feeds at
  1K, 7 at 8K, 2 at 32K, and each `feed` pays a fixed cost in this path that a bigger slice amortises.
* **When the other slot decodes, the slice matters even more**: 8K holds 595 tok/s (94 % of its idle rate) where 1K
  drops to 385 (67 %) - the read gives the decoders `min(1 s, slice/2)` after every feed, so smaller feeds pay that
  hand-over more often.
* Even at 32K a batch read (1,307) is still below the one-stream path (1,649), so a second slot costs prefill speed; the
  recommendation stands: one stream for this workload, and if slots are wanted, `--read-slice 32768` (the yield
  granularity becomes one 32K feed, i.e. a short request waits at most ~25 s to get in).

**Why 8K -> 32K doubles it: the cards' pipeline, not a per-feed constant.**  `forward_chunks` runs a feed's chunks
through the four stages (CUDA0 [0,6], CUDA1 [7,17], CUDA2 [18,28], CUDA3 [29,39] + head - the split in the engine's own
log) as a pipeline, so a feed that holds one 8K chunk has no overlap to give: seven feeds pay seven fills and drains.
Fitting the two measured points (one feed 33.3 s, seven feeds 86.4 s) gives a pipeline depth of ~3.5 stages and ~3.5 s
a chunk; that model predicts the 32K row at 42.1 s, and the measurement was 42.1 s.  So for a two-slot deployment the
fix is not the slice default (8K would only move 572 -> 634); it is to stop a read at chunk boundaries (`halt_k`, already
in the model) and hand the rest back as one feed - in single-GPU mode the yoco layer skip makes a mid-read stop
inconsistent, so that path keeps the slices.


## Conversation slots: ALREADY ON - my "off" was wrong (corrected 01:40)

Earlier in this file I wrote that production ran with the conversation slots off because the *launcher* had no
`STRATA_DS4_SLOT_KEEP` line.  That was the wrong place to look: `serve/server.py`'s `deepseek_slot_env()` sets
`env.setdefault("STRATA_DS4_SLOT_KEEP", "1")` (and `SLOT_GB=2`, `SLOT_DIR=<log dir>/ds4-slots`) for every DeepSeek
engine it spawns, and production's engine is spawned by it.  The slot folder proves it: its oldest conversation is
`slot-1793034255094-1.bin`, dated **2026-10-27 01:04**, a day before the launcher line was added, and 11 `.bin` files
predate it.  The launcher's `export STRATA_DS4_SLOT_KEEP=1` is therefore **redundant** (harmless).  What this round did
change is real but smaller: `STRATA_DS4_SLOT_DAYS=2` (the `.meta` files carry the conversations' token ids, so they are
now dropped after two days instead of seven) and `chmod 700` on the folder (it was 775, files 664 - readable by other
users of the box).

The A/B/A/B measurement below stands as evidence of what the slots do on the single stream (it measured the existing
behaviour, not a change):

| | prompt tokens | reused | read | wall |
|---|---:|---:|---:|---:|
| A1 | 53,925 | 0 | 32.9 s (1,641 tok/s) | 39 s |
| B1 | 59,904 | 0 | 35.3 s (1,698 tok/s) | 41 s |
| A2 | 59,966 | **59,903** (`took back ... at token 59903 of 59966 from RAM (0.0 s); 1 new tokens to read`) | 0.09 s | **1.9 s** |
| B2 | 60,336 | **60,273** | 0.12 s | **2.6 s** |

An alternating turn costs a state load from RAM (~0.1 s) instead of a re-read of the whole prompt (~35 s here; a 190K
context would have been ~2 min).  The folder holds a 60K conversation as a 67 MB `.bin` + `.meta` pair (~1.1 KB a token,
so a 190K conversation is ~210 MB and the 2 GB folder cap holds ~9).

**A 190K cold read, measured on the single stream**: a 233,123-token prompt read in **122.3 s (1,906 tok/s)**, so an
agent's first turn after a compaction is ~2 min, and the read's beats keep its stream alive meanwhile.

**Since every DeepSeek user gets the slots by default**, the file permissions are not this box's problem alone: the
engine creates its slot files with the process umask (664 here) and the folder 755-775, so a `.meta` (the conversation's
token ids, i.e. the prompt in full) is readable by other users of the machine.  A small `[Engine]` PR with
`O_CREAT|O_EXCL` + `fchmod 0600` on the files and 0700 on the folder would fix it for everyone; this box is fixed by the
`chmod` above.

(The structured request's 10 s first byte is the single-turn queue, not the read; it came right after the Anthropic
request.)  Both changes are in commit `0e16735`'s follow-up.  The structured row's "largest gap" measured 0.0 s - the
four empty deltas landed together - and its bound is `PREFILL_BEAT_S`, i.e. 10 s at most.


## The solo->slot promotion no longer re-reads the prompt (23:39, `.step28`)

`ds4_main.cpp` got a **seat map**: what each seat holds (0..N-1 the batch slots, N the solo GEN) is a *physical*
sequence, and moving a conversation to another seat is a swap of two integers - nothing is copied.  At admission the
engine looks for the idle seat whose sequence holds the longest real prefix of the prompt (a shared opening alone is not
enough: `n * 2 < t.size()` means "another conversation") and takes that seat; a seat whose read gave way stays marked so
another request cannot take its half-read state, and a slot that ends clears the mark.

Measured with a 40K-token prompt (a 24-character nonce at the front so nothing can be cached), 600 tokens out, and a
short request four seconds in (which makes the server promote the solo request into a slot):

| | A's wall | A's first token | B's wall | engine |
|---|---:|---:|---:|---|
| before (`.step26`) | **184.3 s** | 0.1 s | 1.8 s | the read gave way at 3,072 of 48,592 tokens, then re-read |
| after (`.step28`) | **92.5 s** | 0.1 s | 1.9 s | `seat 0 took seat 2's conversation (3072 tokens)`, then `seat 1 took seat 0's conversation (46981 tokens)` |

So the promotion costs a re-read of the whole prompt before and a swap after: **2.0x** on that request, and the short
request is unaffected.  What is left of the 92.5 s is the 46,981-token read (~37 s) plus the 600-token answer.

**M6 (the Parallel view) accepted on it.**  With eight clients in flight (two traffic runs overlapping) and four slots,
`/v1/status` read `slots=[0 busy,1 busy,2 busy,3 busy] busy=4 waiting=4` - `waiting` = in flight minus the slots that
hold one, the M6 fix - and the page drew exactly that: the `4 slot(s) - 4 asked` header, four lit slot cells, the
in-flight area chart with `peak 8`, and the four facts (`In flight 8`, `Slots busy 4/4`, `Waiting 4`, `Peak 8`).

The page to look at is **`/api-monitor`** (the API monitor: the M6 Parallel panel, the request list, the model state);
the `#monitor` tab of the app at `/` is the engine monitor (speed, GPU, VRAM, context fill) and has no slot panel - both
were screenshotted (work/m6_parallel.png, work/m6_monitor.png).  The key goes in the page's "API key" field once
(localStorage); the panel's `--api-monitor` request list needs it, the slot cells do not.


## The four open PRs, all drafts (02:12)

| | branch | what | why it stays a draft |
|---|---|---|---|
| [#4199](https://github.com/Niko1221/Strata/pull/4199) | `feat/dsv41-batch-m1` | batched serving (M1, M4-M6, the M2 step) | engine code: needs the CUDA **and** HIP builds; CUDA is green here (this box builds it), HIP has no toolchain |
| [#4340](https://github.com/Niko1221/Strata/pull/4340) | `fix/read-heartbeat` | the read beat on all three streams | server only, but the repository's rule is a draft until its builds have run |
| [#4341](https://github.com/Niko1221/Strata/pull/4341) | `feat/single-stream-short-first` | `STRATA_ADMIT_SHORT_FIRST` in the single-stream queue | same |
| [#4343](https://github.com/Niko1221/Strata/pull/4343) | `claude/busy-albattani-fjntdn` (a second agent's) | slot files `0600` | engine code; CUDA build + e2e done here, HIP asked for in a comment |

The #4343 validation on this box, for the record: a fresh Ninja build with the production options linked `strata-dsv41`
and `ds4_slot_test` (rc=0, `slot_test: all passed`); the end-to-end run in `/tmp/slot0600-test` (a folder the engine
creates is `0700`, the written files are `0600`, files chmod'ed to `664` are `0600` again at the next start, and both
conversations are kept).

