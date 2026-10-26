# Batched decoding for DeepSeek-V4.1 (`strata-dsv41`)

Status: **design, partly built**.  The scheduler's CPU half (M1), the chunked-decoding path in the server (Path B), the
engine's batch protocol (BGEN/BADM/BT/BDONE/BSTOP) and N slots that take turns on the one model by parking their states
(`--batch N`, `--slot-burst K`) are implemented and tested; the kernels (M2 varlen attention, M3 batched MoE) are not -
they are what turns the present interleaving of requests into real throughput.  Scoped for the `deepseek41` engine and
the HTTP server that drives it. Numbers are from a 4 x NVIDIA L20 box (184 GB VRAM, PCIe 4.0, no NVLink) running the IST-DAS
2.25 bpw pack (274 GB, 1M context).

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