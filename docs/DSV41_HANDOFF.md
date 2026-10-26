# DeepSeek-V4.1 batching: where the work stands (handoff)

Branch: `feat/dsv41-batch-m1` in /data/works/Strata (18 commits, not pushed).  Design: `docs/DSV41_BATCHING.md`.

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

Validated on the box: mock-engine A/B (8 tokens without continuation, 40 with), 3-client real run (chunked
continuation works, no interleaving yet), `serve.test_server` 352/352.

## Next, in order

1. **Engine BGEN for one slot.**  Speak `BGEN <slot> <max_new>[ keys] <ids>` (and `BGENI`), behave exactly like
   `GEN` for slot 0, then print `batch_slots 1` in the INFO block.  The server then takes its existing
   `generate_batched` path; single-request output must stay byte-identical.  Needs a model window (~4 min).
2. **Multi-slot BGEN.**  Per-sequence KV (M2): use `ds4_batch_kv.hpp`/`ds4_batch_plan.hpp`, then slots hold
   their state in memory and the serve loop runs one step per slot per turn.  This is what makes three clients
   interleave; the server side already supports it.
3. **varlen attention** (`ds4_attn_split.cuh`, `ds4_model.cu`): per-run block tables instead of one ring.
4. **Batched MoE** (M3): concatenate the routing rows of the slots and run the grouped GEMM once.
5. **Tuning** (M5): DSpark with a batch (likely off), chunk sizes, `batch_slots` count.

## Facts worth keeping

* `serve/server.py:1141`: `self.batch = int(self.info.get("batch_slots") or 0)` - the server's batched path keys
  off the engine's INFO; advertising it before BGEN works breaks every request.
* `serve/server.py:1772` `generate_batched` sends `BGEN <slot> ...` / `BGENI <slot> ...`, reads `PP`/`T`/`DONE`,
  and `STOP <slot>` cancels one slot.
* The server holds `self.fifo` for a whole request; the chunked path alone therefore does not interleave.
* A conversation state (`Model::slot_save/load`) includes the window rings, the compressors' groups, the
  compressed KV and the index keys: its size grows with the context, which is why bursts are coarse.

Production on 10.188.27.29 runs the stock binary (no burst env) and is healthy; restarts are allowed on request.
