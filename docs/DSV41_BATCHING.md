# Batched decoding for DeepSeek-V4.1 (`strata-dsv41`)

Status: **design proposal** (no code yet). Scoped for the `deepseek41` engine and the HTTP server that
drives it. Numbers are from a 4 x NVIDIA L20 box (184 GB VRAM, PCIe 4.0, no NVLink) running the IST-DAS
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
