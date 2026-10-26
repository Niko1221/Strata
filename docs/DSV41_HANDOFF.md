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

## Next, in order

1. **Per-sequence KV (M2)**, which is what turns the present turn-taking into throughput: `ds4_batch_kv.hpp` +
   `ds4_batch_plan.hpp` are the CPU half; the serve loop's `Run`s and the round robin are the place they attach.
   `varlen attention` (`ds4_attn_split.cuh`, `ds4_model.cu`) is the kernel: per-run block tables instead of one ring.
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