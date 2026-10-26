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

Then the two-slot version, same day (18:17-18:24, `--batch 2 --slot-burst 8`): the probe's two admissions at once - each
slot's tokens equal its GEN reference and their BT lines interleave (`0001110000111100001111`); over HTTP three clients
interleave their answer tokens (8 handovers over 47 tokens, 2 with one slot) and the reference prompt admitted through
BGEN behind them is byte-identical to production.  The wall time is the same (4.13 s one slot vs 4.23 s two, 120/60/40
tokens): parking states buys latency fairness, not throughput - that is M2/M3's job.

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