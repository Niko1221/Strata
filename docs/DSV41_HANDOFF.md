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

## Next, in order

1. **Multi-slot BGEN (N > 1).**  Per-sequence KV (M2): use `ds4_batch_kv.hpp`/`ds4_batch_plan.hpp`, then slots hold
   their state in memory and the serve loop runs one step per slot per turn.  This is what makes three clients
   interleave; the server side already supports it (`--batch N` in the engine's args -> `INFO batch_slots=N`).  The
   cheap way there is the parked states of Path B (`ds4_state_bank.hpp`): a burst of K tokens per slot per turn.
2. **varlen attention** (`ds4_attn_split.cuh`, `ds4_model.cu`): per-run block tables instead of one ring.
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
* One slot at a time: the model holds one conversation, so a slot's answer runs to its end before the next request is
  read; the server may run a solo request while a slot is busy, and the engine serves them in turn.
* Test assets on the box: `build/strata-dsv41.batch` (the `--batch` engine), `strata-batch1.json` (production config
  with `--batch 1`, port 8099, `exe` = the .batch binary), `/data/works/batch_window.sh` (probe + server check),
  `/data/works/batch_http_window.sh` (server check alone), logs in `/data/works/batch_*window.log`.  Production's
  `build/strata-dsv41` is kept as the stock binary (`build/strata-dsv41.prod`).
* A mock server from an earlier session (`python -m serve.server --engine mock --port 8099`) was answering `/health`
  200 on 8099 and made the first server check talk to it; it is gone, and `batch_http_window.sh` refuses to start if
  8099 is taken.
* Production on 10.188.27.29 runs the stock binary (no burst env, no `--batch`) and is healthy; restarts are allowed
  on request.
